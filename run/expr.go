package run

import (
	"fmt"

	"m31labs.dev/elio/ir"
)

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
	case "min", "max":
		a, err := ev.eval(c.Args[0])
		if err != nil {
			return nil, err
		}
		b, err := ev.eval(c.Args[1])
		if err != nil {
			return nil, err
		}
		if (c.Func == "min") == (toFloat(a) <= toFloat(b)) {
			return a, nil
		}
		return b, nil
	}
	if v, handled, err := ev.mathBuiltin(c); handled {
		return v, err
	}
	if n, _, ok := ir.VecConstructor(c.Func); ok {
		return ev.vecConstruct(n, c.Args)
	}
	if elem, ok := ir.ScalarCast(c.Func); ok {
		v, err := ev.eval(c.Args[0])
		if err != nil {
			return nil, err
		}
		f := toFloat(v)
		switch elem.Name {
		case "i32", "u32":
			return int64(f), nil // truncate toward zero, matching GPU int conversion
		}
		return f, nil
	}
	return nil, fmt.Errorf("run: unsupported call %q", c.Func)
}

// vecConstruct builds an n-component vector from constructor arguments,
// flattening vector args (so vec4(xyz, w) works) and splatting a single scalar
// (so vec3(0.0) works) — mirroring WGSL/GLSL/Metal constructor semantics.
func (ev *evaluator) vecConstruct(n int, args []ir.Expr) (any, error) {
	out := make([]float64, 0, n)
	for _, a := range args {
		v, err := ev.eval(a)
		if err != nil {
			return nil, err
		}
		if vec, ok := v.([]float64); ok {
			out = append(out, vec...)
		} else {
			out = append(out, toFloat(v))
		}
	}
	if len(out) == 1 && n > 1 { // splat
		s := out[0]
		out = make([]float64, n)
		for i := range out {
			out[i] = s
		}
	}
	if len(out) != n {
		return nil, fmt.Errorf("run: vec%d constructor got %d components", n, len(out))
	}
	return out, nil
}

func (ev *evaluator) addrOf(e ir.Expr) (any, error) {
	switch x := e.(type) {
	case ir.Name:
		v, ok := ev.lookup(x.Name)
		if !ok {
			return nil, fmt.Errorf("run: undefined name %q", x.Name)
		}
		return v, nil
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
