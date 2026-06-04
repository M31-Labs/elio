// Package run is Elio's CPU fallback: a scalar interpreter that executes a
// compute ir.Module on the host, one global invocation at a time.
//
// It is the backend for surfaces with no GPU compute — WebGL and Android
// GLES2, the floor of GoSX's reach — so the same kernel that emits to WGSL/MSL
// also runs here with no device. (GoSX hand-writes such CPU paths today, e.g.
// for particles; Elio generates one from the same IR.) It doubles as the
// execution oracle for the emitter tests: a kernel can be both validated as
// WGSL and run for correctness here.
package run

import (
	"fmt"
	"sync"

	"m31labs.dev/elio/ir"
)

// Memory holds the named resources a kernel reads and writes. Conventions:
//   - storage buffers of structs/records → []any (elements are map[string]any)
//   - vectors → []float64; matrices → Mat (column-major)
//   - uniform blocks → map[string]any
//   - atomic storage arrays → []float64, mutated in place
type Memory struct {
	Vars map[string]any
}

// Mat is a column-major matrix value.
type Mat struct {
	Cols, Rows int
	E          []float64
}

// scalarRef points at a mutable scalar cell (an atomic array element).
type scalarRef struct {
	arr []float64
	idx int
}

type flow int

const (
	flowNormal flow = iota
	flowBreak
	flowReturn
)

// Run executes kernel kernelName over global invocations [0,count) against mem.
// Invocations run sequentially; atomics are therefore exact.
func Run(m *ir.Module, kernelName string, count int, mem *Memory) error {
	var k *ir.Kernel
	for i := range m.Kernels {
		if m.Kernels[i].Name == kernelName {
			k = &m.Kernels[i]
			break
		}
	}
	if k == nil {
		return fmt.Errorf("run: kernel %q not found", kernelName)
	}
	consts, err := evalConsts(m, mem)
	if err != nil {
		return err
	}
	total := k.WorkgroupSize[0] * k.WorkgroupSize[1] * k.WorkgroupSize[2]
	if total < 1 {
		total = 1
	}
	// Embarrassingly-parallel kernels (no shared memory) run one invocation at a
	// time. Kernels with shared memory + barriers need a cooperating workgroup,
	// so each workgroup's invocations run in lockstep goroutines synchronized by
	// a barrier — the CPU fallback then executes (and the race detector audits)
	// reductions and scans, not just maps.
	if len(k.Shared) == 0 {
		for gid := 0; gid < count; gid++ {
			ev := &evaluator{mem: mem, locals: map[string]any{}, consts: consts}
			setBuiltins(ev, k, gid, total)
			if _, err := ev.execBlock(k.Body); err != nil {
				return fmt.Errorf("run: invocation %d: %w", gid, err)
			}
		}
		return nil
	}

	if count%total != 0 {
		return fmt.Errorf("run: shared-memory kernel needs count (%d) to be a multiple of the workgroup size (%d)", count, total)
	}
	for wg := 0; wg < count/total; wg++ {
		shared := initShared(k.Shared)
		bar := newBarrier(total)
		var wgrp sync.WaitGroup
		errs := make([]error, total)
		for l := 0; l < total; l++ {
			wgrp.Add(1)
			go func(l int) {
				defer wgrp.Done()
				gid := wg*total + l
				ev := &evaluator{mem: mem, locals: map[string]any{}, shared: shared, consts: consts, barrier: bar}
				setBuiltins(ev, k, gid, total)
				_, errs[l] = ev.execBlock(k.Body)
			}(l)
		}
		wgrp.Wait()
		for l, err := range errs {
			if err != nil {
				return fmt.Errorf("run: workgroup %d lane %d: %w", wg, l, err)
			}
		}
	}
	return nil
}

// evalConsts evaluates the module's constants once, in order, so each may
// reference earlier ones. The result is shared read-only by every invocation.
func evalConsts(m *ir.Module, mem *Memory) (map[string]any, error) {
	consts := map[string]any{}
	cev := &evaluator{mem: mem, locals: map[string]any{}, consts: consts}
	for _, cn := range m.Consts {
		v, err := cev.eval(cn.Value)
		if err != nil {
			return nil, fmt.Errorf("run: const %q: %w", cn.Name, err)
		}
		consts[cn.Name] = v
	}
	return consts, nil
}

// setBuiltins binds a kernel's @builtin inputs for the invocation at global
// index gid, deriving the local index and workgroup id from the (1-D) workgroup
// size — so local_invocation_id and workgroup_id are available too, not only
// global_invocation_id.
func setBuiltins(ev *evaluator, k *ir.Kernel, gid, total int) {
	lid := gid % total
	wid := gid / total
	// Builtin ids are u32 vectors, so their components are integers — lane-index
	// arithmetic (e.g. the bitonic sort's t/j, t%2) must divide and modulo as
	// integers, not floats.
	for _, bi := range k.Builtins {
		switch bi.Builtin {
		case "global_invocation_id":
			ev.locals[bi.Name] = []int64{int64(gid), 0, 0}
		case "local_invocation_id":
			ev.locals[bi.Name] = []int64{int64(lid), 0, 0}
		case "workgroup_id":
			ev.locals[bi.Name] = []int64{int64(wid), 0, 0}
		case "local_invocation_index":
			ev.locals[bi.Name] = int64(lid)
		}
	}
}

// initShared allocates a workgroup's shared store: one zeroed slice per array
// declaration (scalars start at zero), shared by every lane in the workgroup.
func initShared(decls []ir.Shared) map[string]any {
	shared := map[string]any{}
	for _, sh := range decls {
		if arr, ok := sh.Type.(ir.Array); ok && arr.Len > 0 {
			shared[sh.Name] = make([]float64, arr.Len)
		} else {
			shared[sh.Name] = float64(0)
		}
	}
	return shared
}

// barrier is a single-use-per-phase cyclic barrier for the lanes of one
// workgroup. wait blocks until all `n` lanes arrive, then releases them
// together; its mutex establishes the happens-before that makes shared writes
// before a barrier visible to reads after it.
type barrier struct {
	mu         sync.Mutex
	cond       *sync.Cond
	n          int
	count      int
	generation int
}

func newBarrier(n int) *barrier {
	b := &barrier{n: n}
	b.cond = sync.NewCond(&b.mu)
	return b
}

func (b *barrier) wait() {
	b.mu.Lock()
	defer b.mu.Unlock()
	gen := b.generation
	b.count++
	if b.count == b.n {
		b.count = 0
		b.generation++
		b.cond.Broadcast()
		return
	}
	for gen == b.generation {
		b.cond.Wait()
	}
}

type evaluator struct {
	mem     *Memory
	locals  map[string]any
	shared  map[string]any // workgroup-shared store (nil for non-shared kernels)
	consts  map[string]any // module-level constants (shared, read-only)
	barrier *barrier       // workgroup barrier (nil for non-shared kernels)
}

func (ev *evaluator) lookup(name string) (any, bool) {
	if v, ok := ev.locals[name]; ok {
		return v, true
	}
	if v, ok := ev.shared[name]; ok {
		return v, true
	}
	if v, ok := ev.consts[name]; ok {
		return v, true
	}
	v, ok := ev.mem.Vars[name]
	return v, ok
}
