package run

import (
	"math"
	"testing"

	"m31labs.dev/elio/ir"
)

// TestRunMathBuiltins executes a kernel exercising the scalar and vector math
// builtins on the CPU fallback and checks each result, proving the interpreter
// is the executing reference the GPU backends are cross-validated against.
func TestRunMathBuiltins(t *testing.T) {
	lit := func(s string) ir.Expr { return ir.Lit{Text: s} }
	vec3 := func(a, b, c string) ir.Expr {
		return ir.Call{Func: "vec3", Args: []ir.Expr{lit(a), lit(b), lit(c)}}
	}
	call := func(name string, args ...ir.Expr) ir.Expr { return ir.Call{Func: name, Args: args} }
	set := func(i int, v ir.Expr) ir.Stmt {
		return ir.Assign{Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Lit{Text: itoa(i)}}, Value: v}
	}

	mod := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.F32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				set(0, call("length", vec3("3.0", "4.0", "0.0"))),                                                     // 5
				set(1, call("mix", lit("0.0"), lit("10.0"), lit("0.5"))),                                              // 5
				set(2, call("clamp", lit("5.0"), lit("0.0"), lit("1.0"))),                                             // 1
				set(3, ir.Member{E: call("cross", vec3("1.0", "0.0", "0.0"), vec3("0.0", "1.0", "0.0")), Field: "z"}), // 1
				set(4, ir.Member{E: call("normalize", vec3("0.0", "3.0", "0.0")), Field: "y"}),                        // 1
				set(5, call("sqrt", lit("16.0"))),                                                                     // 4
				set(6, call("fract", lit("2.25"))),                                                                    // 0.25
				set(7, call("dot", vec3("1.0", "2.0", "3.0"), vec3("4.0", "5.0", "6.0"))),                             // 32
			},
		}},
	}
	acc := make([]float64, 8)
	if err := Run(mod, "main", 1, &Memory{Vars: map[string]any{"acc": acc}}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := []float64{5, 5, 1, 1, 1, 4, 0.25, 32}
	for i, w := range want {
		if math.Abs(acc[i]-w) > 1e-9 {
			t.Errorf("acc[%d] = %v, want %v", i, acc[i], w)
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}
