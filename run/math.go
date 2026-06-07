package run

import (
	"fmt"
	"math"

	"m31labs.dev/elio/ir"
)

// mathBuiltins is the set of shader math builtins the CPU fallback implements.
// They are spelled identically on the GPU backends (WGSL/GLSL/Metal emit them
// verbatim); this is the executing reference the conformance suite compares
// against. Membership-gated so non-math calls fall through to the rest of the
// dispatch (vector constructors, atomics, error).
var mathBuiltins = map[string]bool{
	"sin": true, "cos": true, "tan": true, "sqrt": true, "exp": true,
	"log": true, "exp2": true, "log2": true, "floor": true, "ceil": true,
	"fract": true, "abs": true, "sign": true,
	"pow": true, "step": true, "distance": true, "reflect": true, "cross": true,
	"length": true, "normalize": true,
	"clamp": true, "mix": true, "smoothstep": true,
}

// mathBuiltin evaluates a shader math builtin on the CPU. The bool result
// reports whether the call was a math builtin at all (false → not handled here,
// let the caller continue its dispatch). Scalar functions map component-wise
// over vectors, matching WGSL/GLSL/Metal.
func (ev *evaluator) mathBuiltin(c ir.Call) (any, bool, error) {
	if !mathBuiltins[c.Func] {
		return nil, false, nil
	}
	args := make([]any, len(c.Args))
	for i, a := range c.Args {
		v, err := ev.eval(a)
		if err != nil {
			return nil, true, err
		}
		args[i] = v
	}
	arg := func(i int) any {
		if i < len(args) {
			return args[i]
		}
		return 0.0
	}

	switch c.Func {
	case "sin":
		return mapElem(arg(0), math.Sin), true, nil
	case "cos":
		return mapElem(arg(0), math.Cos), true, nil
	case "tan":
		return mapElem(arg(0), math.Tan), true, nil
	case "sqrt":
		return mapElem(arg(0), math.Sqrt), true, nil
	case "exp":
		return mapElem(arg(0), math.Exp), true, nil
	case "exp2":
		return mapElem(arg(0), math.Exp2), true, nil
	case "log":
		return mapElem(arg(0), math.Log), true, nil
	case "log2":
		return mapElem(arg(0), math.Log2), true, nil
	case "floor":
		return mapElem(arg(0), math.Floor), true, nil
	case "ceil":
		return mapElem(arg(0), math.Ceil), true, nil
	case "abs":
		return mapElem(arg(0), math.Abs), true, nil
	case "fract":
		return mapElem(arg(0), func(x float64) float64 { return x - math.Floor(x) }), true, nil
	case "sign":
		return mapElem(arg(0), func(x float64) float64 {
			switch {
			case x > 0:
				return 1
			case x < 0:
				return -1
			default:
				return 0
			}
		}), true, nil
	case "pow":
		return zip2(arg(0), arg(1), math.Pow), true, nil
	case "step":
		// step(edge, x): x < edge ? 0 : 1
		return zip2(arg(1), arg(0), func(x, edge float64) float64 {
			if x < edge {
				return 0
			}
			return 1
		}), true, nil
	case "length":
		return length64(toVec(arg(0))), true, nil
	case "distance":
		a, b := toVec(arg(0)), toVec(arg(1))
		return length64(sub64(a, b)), true, nil
	case "normalize":
		v := toVec(arg(0))
		l := length64(v)
		if l == 0 {
			return append([]float64(nil), v...), true, nil
		}
		out := make([]float64, len(v))
		for i := range v {
			out[i] = v[i] / l
		}
		return out, true, nil
	case "cross":
		a, b := toVec(arg(0)), toVec(arg(1))
		if len(a) != 3 || len(b) != 3 {
			return nil, true, fmt.Errorf("run: cross requires vec3 operands")
		}
		return []float64{
			a[1]*b[2] - a[2]*b[1],
			a[2]*b[0] - a[0]*b[2],
			a[0]*b[1] - a[1]*b[0],
		}, true, nil
	case "reflect":
		// reflect(i, n) = i - 2*dot(n, i)*n
		i, n := toVec(arg(0)), toVec(arg(1))
		d := dot64(n, i)
		out := make([]float64, len(i))
		for k := range i {
			out[k] = i[k] - 2*d*n[k]
		}
		return out, true, nil
	case "clamp":
		return zip3(arg(0), arg(1), arg(2), func(x, lo, hi float64) float64 {
			return math.Min(math.Max(x, lo), hi)
		}), true, nil
	case "mix":
		// mix(a, b, t) = a + (b-a)*t
		return zip3(arg(0), arg(1), arg(2), func(a, b, t float64) float64 {
			return a + (b-a)*t
		}), true, nil
	case "smoothstep":
		// smoothstep(e0, e1, x)
		return zip3(arg(0), arg(1), arg(2), func(e0, e1, x float64) float64 {
			if e1 == e0 {
				return 0
			}
			t := math.Min(math.Max((x-e0)/(e1-e0), 0), 1)
			return t * t * (3 - 2*t)
		}), true, nil
	}
	return nil, false, nil
}

// mapElem applies f to a scalar or component-wise to a vector ([]float64).
func mapElem(v any, f func(float64) float64) any {
	if vec, ok := v.([]float64); ok {
		out := make([]float64, len(vec))
		for i, x := range vec {
			out[i] = f(x)
		}
		return out
	}
	return f(toFloat(v))
}

// zip2 applies f over two operands, broadcasting a scalar against a vector and
// matching vectors component-wise.
func zip2(a, b any, f func(x, y float64) float64) any {
	av, aVec := a.([]float64)
	bv, bVec := b.([]float64)
	if !aVec && !bVec {
		return f(toFloat(a), toFloat(b))
	}
	n := vecLen(av, aVec, bv, bVec)
	out := make([]float64, n)
	as, bs := toFloat(a), toFloat(b)
	for i := 0; i < n; i++ {
		x, y := as, bs
		if aVec {
			x = av[i]
		}
		if bVec {
			y = bv[i]
		}
		out[i] = f(x, y)
	}
	return out
}

// zip3 applies f over three operands, broadcasting scalars against vectors.
func zip3(a, b, c any, f func(x, y, z float64) float64) any {
	av, aVec := a.([]float64)
	bv, bVec := b.([]float64)
	cv, cVec := c.([]float64)
	if !aVec && !bVec && !cVec {
		return f(toFloat(a), toFloat(b), toFloat(c))
	}
	n := 0
	for _, p := range []struct {
		v  []float64
		ok bool
	}{{av, aVec}, {bv, bVec}, {cv, cVec}} {
		if p.ok && len(p.v) > n {
			n = len(p.v)
		}
	}
	out := make([]float64, n)
	as, bs, cs := toFloat(a), toFloat(b), toFloat(c)
	for i := 0; i < n; i++ {
		x, y, z := as, bs, cs
		if aVec {
			x = av[i]
		}
		if bVec {
			y = bv[i]
		}
		if cVec {
			z = cv[i]
		}
		out[i] = f(x, y, z)
	}
	return out
}

func vecLen(av []float64, aVec bool, bv []float64, bVec bool) int {
	n := 0
	if aVec && len(av) > n {
		n = len(av)
	}
	if bVec && len(bv) > n {
		n = len(bv)
	}
	return n
}

func dot64(a, b []float64) float64 {
	var s float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

func length64(v []float64) float64 { return math.Sqrt(dot64(v, v)) }

func sub64(a, b []float64) []float64 {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		var x, y float64
		if i < len(a) {
			x = a[i]
		}
		if i < len(b) {
			y = b[i]
		}
		out[i] = x - y
	}
	return out
}
