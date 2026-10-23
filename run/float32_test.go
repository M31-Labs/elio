package run

import (
	"math"
	"testing"

	"m31labs.dev/elio/ir"
)

// sumKernel stores 0.1 + 0.2 and sqrt(2.0), and a vec4 built from them, into
// out (a scalar buffer) and v (a vec4 buffer).
func sumKernel() *ir.Module {
	n := func(s string) ir.Expr { return ir.Name{Name: s} }
	idx := func(b string, i string) ir.Expr { return ir.Index{E: n(b), Idx: ir.Lit{Text: i}} }
	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "out", Type: ir.Array{Elem: ir.F32}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "v", Type: ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}}},
		},
		Kernels: []ir.Kernel{{Name: "k", WorkgroupSize: [3]int{1, 1, 1}, Body: []ir.Stmt{
			ir.Assign{Target: idx("out", "0"), Value: ir.Binary{Op: "+", L: ir.Lit{Text: "0.1"}, R: ir.Lit{Text: "0.2"}}},
			ir.Assign{Target: idx("out", "1"), Value: ir.Call{Func: "sqrt", Args: []ir.Expr{ir.Lit{Text: "2.0"}}}},
			ir.Let{Name: "a", Value: ir.Call{Func: "vec4", Args: []ir.Expr{ir.Lit{Text: "0.1"}, ir.Lit{Text: "0.2"}, ir.Lit{Text: "1.0"}, ir.Lit{Text: "3.0"}}}},
			ir.Assign{Target: idx("v", "0"), Value: ir.Binary{Op: "/", L: n("a"), R: ir.Lit{Text: "3.0"}}},
			ir.Assign{Target: idx("out", "2"), Value: ir.Call{Func: "dot", Args: []ir.Expr{n("a"), n("a")}}},
			ir.Assign{Target: idx("out", "3"), Value: ir.Call{Func: "f32", Args: []ir.Expr{ir.Lit{Text: "16777217"}}}},
		}}},
	}
}

func runSum(t *testing.T, opts *Options) ([]float64, []any) {
	t.Helper()
	out := make([]float64, 4)
	v := []any{[]float64{0, 0, 0, 0}}
	mem := &Memory{Vars: map[string]any{"out": out, "v": v}}
	var err error
	if opts == nil {
		err = Run(sumKernel(), "k", 1, mem)
	} else {
		err = RunWith(sumKernel(), "k", 1, mem, *opts)
	}
	if err != nil {
		t.Fatal(err)
	}
	return out, v
}

func TestFloat32ModeRoundsArithmetic(t *testing.T) {
	a, b := float32(0.1), float32(0.2)
	d1, d2 := 0.1, 0.2
	want := float64(a + b)
	if want == d1+d2 {
		t.Fatal("test premise broken: float32 and float64 sums are equal")
	}
	out, v := runSum(t, &Options{Float32: true})
	if out[0] != want {
		t.Errorf("float32 0.1+0.2 = %v, want %v", out[0], want)
	}
	if want := float64(float32(math.Sqrt(2))); out[1] != want {
		t.Errorf("float32 sqrt(2) = %v, want %v", out[1], want)
	}
	f := v[0].([]float64)
	if want := float64(float32(0.1) / float32(3)); f[0] != want {
		t.Errorf("float32 vec 0.1/3 = %v, want %v", f[0], want)
	}
	x, y, z, w := float32(0.1), float32(0.2), float32(1), float32(3)
	if want := float64(x*x + y*y + z*z + w*w); math.Abs(out[2]-want) > 1e-6 {
		t.Errorf("float32 dot = %v, want about %v", out[2], want)
	}
	if out[2] != float64(float32(out[2])) {
		t.Errorf("dot result %v is not a float32 value", out[2])
	}
	if out[3] != 16777216 {
		t.Errorf("f32(16777217) = %v, want 16777216", out[3])
	}
}

func TestRunStaysFloat64(t *testing.T) {
	out, _ := runSum(t, nil)
	d1, d2 := 0.1, 0.2
	if out[0] != d1+d2 {
		t.Errorf("Run 0.1+0.2 = %v, want the float64 sum", out[0])
	}
	if out[1] != math.Sqrt(2) {
		t.Errorf("Run sqrt(2) = %v", out[1])
	}
	zero, _ := runSum(t, &Options{})
	if zero[0] != out[0] {
		t.Errorf("zero Options differs from Run: %v vs %v", zero[0], out[0])
	}
}
