package run

import (
	"fmt"
	"strconv"
	"strings"
)

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
