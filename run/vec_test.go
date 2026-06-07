package run

import (
	"testing"

	"m31labs.dev/elio/ir"
)

// TestRunVecConstructor executes a kernel that builds a vec3 with the new
// constructor, does vector arithmetic (vec * scalar), flattens a vector into a
// larger constructor (vec4(vec3, w)), and reads components back via swizzle —
// proving the CPU fallback supports vector construction end to end.
func TestRunVecConstructor(t *testing.T) {
	lit := func(s string) ir.Expr { return ir.Lit{Text: s} }
	mod := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.F32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				// v = vec3(2, 3, 4)
				ir.Let{Name: "v", Value: ir.Call{Func: "vec3", Args: []ir.Expr{lit("2.0"), lit("3.0"), lit("4.0")}}},
				// w = v * 2.0  => (4, 6, 8)
				ir.Let{Name: "w", Value: ir.Binary{Op: "*", L: ir.Name{Name: "v"}, R: lit("2.0")}},
				// q = vec4(w, 1.0) => (4, 6, 8, 1)  (flatten a vec into a larger ctor)
				ir.Let{Name: "q", Value: ir.Call{Func: "vec4", Args: []ir.Expr{ir.Name{Name: "w"}, lit("1.0")}}},
				// acc[gid.x] = v.x + w.z + q.w  => 2 + 8 + 1 = 11
				ir.Assign{
					Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
					Value: ir.Binary{Op: "+",
						L: ir.Binary{Op: "+",
							L: ir.Member{E: ir.Name{Name: "v"}, Field: "x"},
							R: ir.Member{E: ir.Name{Name: "w"}, Field: "z"}},
						R: ir.Member{E: ir.Name{Name: "q"}, Field: "w"}},
				},
			},
		}},
	}
	acc := []float64{0}
	if err := Run(mod, "main", 1, &Memory{Vars: map[string]any{"acc": acc}}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if acc[0] != 11 {
		t.Fatalf("acc[0] = %v, want 11 (v.x=2 + w.z=8 + q.w=1)", acc[0])
	}
}

// TestRunVecSplat verifies the single-argument splat form vec3(s) = (s,s,s).
func TestRunVecSplat(t *testing.T) {
	mod := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.F32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "v", Value: ir.Call{Func: "vec3", Args: []ir.Expr{ir.Lit{Text: "5.0"}}}},
				ir.Assign{
					Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
					Value: ir.Binary{Op: "+",
						L: ir.Member{E: ir.Name{Name: "v"}, Field: "x"},
						R: ir.Member{E: ir.Name{Name: "v"}, Field: "z"}},
				},
			},
		}},
	}
	acc := []float64{0}
	if err := Run(mod, "main", 1, &Memory{Vars: map[string]any{"acc": acc}}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if acc[0] != 10 {
		t.Fatalf("acc[0] = %v, want 10 (splat 5 -> v.x+v.z)", acc[0])
	}
}
