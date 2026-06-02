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
	// The CPU fallback runs invocations independently, so it cannot model a
	// cooperating workgroup (shared memory + barriers need lockstep execution).
	if len(k.Shared) > 0 {
		return fmt.Errorf("run: kernel %q uses workgroup-shared memory, which the CPU fallback does not support", kernelName)
	}
	for gid := 0; gid < count; gid++ {
		ev := &evaluator{mem: mem, locals: map[string]any{}}
		for _, bi := range k.Builtins {
			if bi.Builtin == "global_invocation_id" {
				ev.locals[bi.Name] = []float64{float64(gid), 0, 0}
			}
		}
		if _, err := ev.execBlock(k.Body); err != nil {
			return fmt.Errorf("run: invocation %d: %w", gid, err)
		}
	}
	return nil
}

type evaluator struct {
	mem    *Memory
	locals map[string]any
}

func (ev *evaluator) lookup(name string) (any, bool) {
	if v, ok := ev.locals[name]; ok {
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
		if err := ev.assign(x.Target, v); err != nil {
			return flowNormal, err
		}
	case ir.Return:
		return flowReturn, nil
	case ir.Break:
		return flowBreak, nil
	case ir.Barrier:
		return flowNormal, fmt.Errorf("run: workgroup barrier requires lockstep execution, unsupported in the CPU fallback")
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
	}
	return nil, fmt.Errorf("run: cannot access .%s on %T", field, base)
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
	f, _ := strconv.ParseFloat(strings.TrimRight(t, "uif"), 64)
	return f
}

func binop(op string, l, r any) (any, error) {
	if isVec(l) || isVec(r) {
		return vecBinop(op, l, r)
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
	}
	return false
}

func toVec(v any) []float64 {
	if vv, ok := v.([]float64); ok {
		return vv
	}
	return nil
}
