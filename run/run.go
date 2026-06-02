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
	"strconv"
	"strings"
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

// --- statements ---

func (ev *evaluator) execBlock(body []ir.Stmt) (flow, error) {
	for _, s := range body {
		f, err := ev.execStmt(s)
		if err != nil {
			return flowNormal, err
		}
		if f != flowNormal {
			return f, nil
		}
	}
	return flowNormal, nil
}

func (ev *evaluator) execStmt(s ir.Stmt) (flow, error) {
	switch x := s.(type) {
	case ir.Let:
		v, err := ev.eval(x.Value)
		if err != nil {
			return flowNormal, err
		}
		ev.locals[x.Name] = v
	case ir.Var:
		v, err := ev.eval(x.Init)
		if err != nil {
			return flowNormal, err
		}
		ev.locals[x.Name] = v
	case ir.Assign:
		v, err := ev.eval(x.Value)
		if err != nil {
			return flowNormal, err
		}
		if x.Op != "" { // compound assignment: target = target Op value
			cur, err := ev.eval(x.Target)
			if err != nil {
				return flowNormal, err
			}
			if v, err = binop(x.Op, cur, v); err != nil {
				return flowNormal, err
			}
		}
		if err := ev.assign(x.Target, v); err != nil {
			return flowNormal, err
		}
	case ir.Return:
		return flowReturn, nil
	case ir.Break:
		return flowBreak, nil
	case ir.Barrier:
		if ev.barrier == nil {
			return flowNormal, fmt.Errorf("run: barrier outside a workgroup (kernel declares no shared memory)")
		}
		ev.barrier.wait()
	case ir.Do:
		_, err := ev.eval(x.Expr)
		return flowNormal, err
	case ir.If:
		c, err := ev.eval(x.Cond)
		if err != nil {
			return flowNormal, err
		}
		if toBool(c) {
			return ev.execBlock(x.Then)
		}
		if len(x.Else) > 0 {
			return ev.execBlock(x.Else)
		}
	case ir.For:
		if x.Init != nil {
			if _, err := ev.execStmt(x.Init); err != nil {
				return flowNormal, err
			}
		}
		for {
			c, err := ev.eval(x.Cond)
			if err != nil {
				return flowNormal, err
			}
			if !toBool(c) {
				break
			}
			f, err := ev.execBlock(x.Body)
			if err != nil {
				return flowNormal, err
			}
			if f == flowReturn {
				return flowReturn, nil
			}
			if f == flowBreak {
				break
			}
			if x.Post != nil {
				if _, err := ev.execStmt(x.Post); err != nil {
					return flowNormal, err
				}
			}
		}
	case ir.While:
		for {
			c, err := ev.eval(x.Cond)
			if err != nil {
				return flowNormal, err
			}
			if !toBool(c) {
				break
			}
			f, err := ev.execBlock(x.Body)
			if err != nil {
				return flowNormal, err
			}
			if f == flowReturn {
				return flowReturn, nil
			}
			if f == flowBreak {
				break
			}
		}
	default:
		return flowNormal, fmt.Errorf("run: unsupported statement %T", s)
	}
	return flowNormal, nil
}

func (ev *evaluator) assign(target ir.Expr, v any) error {
	switch t := target.(type) {
	case ir.Name:
		ev.locals[t.Name] = v
		return nil
	case ir.Index:
		base, err := ev.eval(t.E)
		if err != nil {
			return err
		}
		idx, err := ev.eval(t.Idx)
		if err != nil {
			return err
		}
		i := int(toFloat(idx))
		switch arr := base.(type) {
		case []any:
			if i < 0 || i >= len(arr) {
				return fmt.Errorf("run: index %d out of range", i)
			}
			arr[i] = v
		case []float64:
			if i < 0 || i >= len(arr) {
				return fmt.Errorf("run: index %d out of range", i)
			}
			arr[i] = toFloat(v)
		default:
			return fmt.Errorf("run: cannot index-assign into %T", base)
		}
		return nil
	case ir.Member:
		// e.g. particles[i].pos = …  — resolve the struct and set its field.
		base, err := ev.eval(t.E)
		if err != nil {
			return err
		}
		m, ok := base.(map[string]any)
		if !ok {
			return fmt.Errorf("run: cannot assign to .%s of %T", t.Field, base)
		}
		m[t.Field] = v
		return nil
	}
	return fmt.Errorf("run: unsupported assign target %T", target)
}

// --- expressions ---

func (ev *evaluator) eval(e ir.Expr) (any, error) {
	switch x := e.(type) {
	case ir.Name:
		v, ok := ev.lookup(x.Name)
		if !ok {
			return nil, fmt.Errorf("run: undefined name %q", x.Name)
		}
		return v, nil
	case ir.Lit:
		return litValue(x.Text), nil
	case ir.Unary:
		v, err := ev.eval(x.E)
		if err != nil {
			return nil, err
		}
		switch x.Op {
		case "-":
			return -toFloat(v), nil
		case "!":
			return !toBool(v), nil
		}
		return nil, fmt.Errorf("run: unsupported unary %q", x.Op)
	case ir.Binary:
		l, err := ev.eval(x.L)
		if err != nil {
			return nil, err
		}
		r, err := ev.eval(x.R)
		if err != nil {
			return nil, err
		}
		return binop(x.Op, l, r)
	case ir.Call:
		return ev.call(x)
	case ir.Index:
		base, err := ev.eval(x.E)
		if err != nil {
			return nil, err
		}
		idx, err := ev.eval(x.Idx)
		if err != nil {
			return nil, err
		}
		return indexValue(base, int(toFloat(idx)))
	case ir.Member:
		base, err := ev.eval(x.E)
		if err != nil {
			return nil, err
		}
		return memberValue(base, x.Field)
	case ir.AddrOf:
		return ev.addrOf(x.E)
	}
	return nil, fmt.Errorf("run: unsupported expression %T", e)
}

func (ev *evaluator) call(c ir.Call) (any, error) {
	switch c.Func {
	case "arrayLength":
		v, err := ev.eval(c.Args[0])
		if err != nil {
			return nil, err
		}
		if arr, ok := v.([]any); ok {
			return float64(len(arr)), nil
		}
		if arr, ok := v.([]float64); ok {
			return float64(len(arr)), nil
		}
		return nil, fmt.Errorf("run: arrayLength on non-array %T", v)
	case "dot":
		a, err := ev.eval(c.Args[0])
		if err != nil {
			return nil, err
		}
		b, err := ev.eval(c.Args[1])
		if err != nil {
			return nil, err
		}
		va, vb := toVec(a), toVec(b)
		if len(va) != len(vb) {
			return nil, fmt.Errorf("run: dot of mismatched vectors")
		}
		sum := 0.0
		for i := range va {
			sum += va[i] * vb[i]
		}
		return sum, nil
	case "atomicAdd":
		ref, err := ev.eval(c.Args[0])
		if err != nil {
			return nil, err
		}
		add, err := ev.eval(c.Args[1])
		if err != nil {
			return nil, err
		}
		sr, ok := ref.(*scalarRef)
		if !ok {
			return nil, fmt.Errorf("run: atomicAdd needs an atomic pointer, got %T", ref)
		}
		old := sr.arr[sr.idx]
		sr.arr[sr.idx] = old + toFloat(add)
		return old, nil
	}
	return nil, fmt.Errorf("run: unsupported call %q", c.Func)
}

func (ev *evaluator) addrOf(e ir.Expr) (any, error) {
	switch x := e.(type) {
	case ir.Name:
		v, ok := ev.lookup(x.Name)
		if !ok {
			return nil, fmt.Errorf("run: undefined name %q", x.Name)
		}
		return v, nil // &array → the array, for arrayLength
	case ir.Index:
		base, err := ev.eval(x.E)
		if err != nil {
			return nil, err
		}
		idx, err := ev.eval(x.Idx)
		if err != nil {
			return nil, err
		}
		if af, ok := base.([]float64); ok {
			return &scalarRef{arr: af, idx: int(toFloat(idx))}, nil
		}
		return nil, fmt.Errorf("run: &index on non-atomic array %T", base)
	}
	return nil, fmt.Errorf("run: unsupported address-of %T", e)
}

// --- value helpers ---

func indexValue(base any, i int) (any, error) {
	switch b := base.(type) {
	case []any:
		if i < 0 || i >= len(b) {
			return nil, fmt.Errorf("run: index %d out of range", i)
		}
		return b[i], nil
	case []float64:
		if i < 0 || i >= len(b) {
			return nil, fmt.Errorf("run: index %d out of range", i)
		}
		return b[i], nil
	case Mat:
		if i < 0 || i >= b.Cols {
			return nil, fmt.Errorf("run: matrix column %d out of range", i)
		}
		col := make([]float64, b.Rows)
		for r := 0; r < b.Rows; r++ {
			col[r] = b.E[i*b.Rows+r]
		}
		return col, nil
	}
	return nil, fmt.Errorf("run: cannot index %T", base)
}

func memberValue(base any, field string) (any, error) {
	switch b := base.(type) {
	case map[string]any:
		v, ok := b[field]
		if !ok {
			return nil, fmt.Errorf("run: no field %q", field)
		}
		return v, nil
	case []float64:
		return swizzle(b, field)
	case []int64:
		return intSwizzle(b, field)
	}
	return nil, fmt.Errorf("run: cannot access .%s on %T", field, base)
}

// intSwizzle is swizzle for integer vectors (builtin ids): a single component
// returns an int64; multiple components return an []int64.
func intSwizzle(v []int64, field string) (any, error) {
	idx := func(c byte) int {
		switch c {
		case 'x', 'r':
			return 0
		case 'y', 'g':
			return 1
		case 'z', 'b':
			return 2
		case 'w', 'a':
			return 3
		}
		return -1
	}
	if len(field) == 1 {
		i := idx(field[0])
		if i < 0 || i >= len(v) {
			return nil, fmt.Errorf("run: bad swizzle .%s", field)
		}
		return v[i], nil
	}
	out := make([]int64, len(field))
	for k := 0; k < len(field); k++ {
		i := idx(field[k])
		if i < 0 || i >= len(v) {
			return nil, fmt.Errorf("run: bad swizzle .%s", field)
		}
		out[k] = v[i]
	}
	return out, nil
}

func swizzle(v []float64, field string) (any, error) {
	component := func(c byte) (int, bool) {
		switch c {
		case 'x', 'r':
			return 0, true
		case 'y', 'g':
			return 1, true
		case 'z', 'b':
			return 2, true
		case 'w', 'a':
			return 3, true
		}
		return 0, false
	}
	if len(field) == 1 {
		i, ok := component(field[0])
		if !ok || i >= len(v) {
			return nil, fmt.Errorf("run: bad swizzle .%s", field)
		}
		return v[i], nil
	}
	out := make([]float64, len(field))
	for k := 0; k < len(field); k++ {
		i, ok := component(field[k])
		if !ok || i >= len(v) {
			return nil, fmt.Errorf("run: bad swizzle .%s", field)
		}
		out[k] = v[i]
	}
	return out, nil
}

func litValue(t string) any {
	switch t {
	case "true":
		return true
	case "false":
		return false
	}
	// Float if it has a decimal point or an explicit 'f' suffix; otherwise an
	// integer (u32 / i32 literal), so that integer division and modulo truncate
	// exactly as every GPU backend does.
	if strings.Contains(t, ".") || strings.HasSuffix(t, "f") {
		f, _ := strconv.ParseFloat(strings.TrimRight(t, "uif"), 64)
		return f
	}
	n, _ := strconv.ParseInt(strings.TrimRight(t, "uif"), 10, 64)
	return n
}

func binop(op string, l, r any) (any, error) {
	if isVec(l) || isVec(r) {
		return vecBinop(op, l, r)
	}
	// Integer operands (u32 / i32) use integer arithmetic — crucially, `/` and
	// `%` truncate, so a reduction stride 32→16→…→1→0 terminates instead of
	// halving forever as floats.
	if li, ok := l.(int64); ok {
		if ri, ok := r.(int64); ok {
			return intBinop(op, li, ri)
		}
	}
	a, b := toFloat(l), toFloat(r)
	switch op {
	case "+":
		return a + b, nil
	case "-":
		return a - b, nil
	case "*":
		return a * b, nil
	case "/":
		return a / b, nil
	case "<":
		return a < b, nil
	case ">":
		return a > b, nil
	case "<=":
		return a <= b, nil
	case ">=":
		return a >= b, nil
	case "==":
		return a == b, nil
	case "!=":
		return a != b, nil
	}
	return nil, fmt.Errorf("run: unsupported binary operator %q", op)
}

func intBinop(op string, a, b int64) (any, error) {
	switch op {
	case "+":
		return a + b, nil
	case "-":
		return a - b, nil
	case "*":
		return a * b, nil
	case "/":
		if b == 0 {
			return nil, fmt.Errorf("run: integer divide by zero")
		}
		return a / b, nil
	case "%":
		if b == 0 {
			return nil, fmt.Errorf("run: integer modulo by zero")
		}
		return a % b, nil
	case "<":
		return a < b, nil
	case ">":
		return a > b, nil
	case "<=":
		return a <= b, nil
	case ">=":
		return a >= b, nil
	case "==":
		return a == b, nil
	case "!=":
		return a != b, nil
	}
	return nil, fmt.Errorf("run: unsupported integer operator %q", op)
}

func isVec(v any) bool { _, ok := v.([]float64); return ok }

// vecBinop applies an arithmetic operator component-wise, broadcasting a scalar
// operand across the vector (vec*vec, vec+vec, vec*scalar, scalar*vec).
func vecBinop(op string, l, r any) (any, error) {
	lv, lvec := l.([]float64)
	rv, rvec := r.([]float64)
	n := 0
	if lvec {
		n = len(lv)
	}
	if rvec {
		if lvec && len(rv) != n {
			return nil, fmt.Errorf("run: vector size mismatch %d vs %d", len(lv), len(rv))
		}
		n = len(rv)
	}
	ls, rs := toFloat(l), toFloat(r)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		a, b := ls, rs
		if lvec {
			a = lv[i]
		}
		if rvec {
			b = rv[i]
		}
		switch op {
		case "+":
			out[i] = a + b
		case "-":
			out[i] = a - b
		case "*":
			out[i] = a * b
		case "/":
			out[i] = a / b
		default:
			return nil, fmt.Errorf("run: operator %q not defined on vectors", op)
		}
	}
	return out, nil
}

func toFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int64:
		return float64(x)
	case bool:
		if x {
			return 1
		}
	}
	return 0
}

func toBool(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case float64:
		return x != 0
	case int64:
		return x != 0
	}
	return false
}

func toVec(v any) []float64 {
	if vv, ok := v.([]float64); ok {
		return vv
	}
	return nil
}
