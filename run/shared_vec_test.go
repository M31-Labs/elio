package run

import (
	"testing"

	"m31labs.dev/elio/ir"
)

// TestSharedVectorArray pins that a workgroup-shared array of vec4 holds whole
// vectors: each lane stores a vector, and after a barrier reads its neighbour's.
func TestSharedVectorArray(t *testing.T) {
	nm := func(s string) ir.Expr { return ir.Name{Name: s} }
	lid := ir.Member{E: nm("lid"), Field: "x"}
	v4 := ir.Vec{N: 4, Elem: ir.F32}
	m := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "out", Type: ir.Array{Elem: v4}}},
		Kernels: []ir.Kernel{{
			Name:          "k",
			WorkgroupSize: [3]int{4, 1, 1},
			Builtins:      []ir.Builtin{{Name: "lid", Builtin: "local_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Shared:        []ir.Shared{{Name: "sh", Type: ir.Array{Elem: v4, Len: 4}}},
			Body: []ir.Stmt{
				ir.Assign{Target: ir.Index{E: nm("sh"), Idx: lid}, Value: ir.Call{Func: "vec4", Args: []ir.Expr{
					ir.Call{Func: "f32", Args: []ir.Expr{lid}}, ir.Lit{Text: "1.0"}, ir.Lit{Text: "2.0"}, ir.Lit{Text: "3.0"}}}},
				ir.Barrier{},
				ir.Assign{Target: ir.Index{E: nm("out"), Idx: lid},
					Value: ir.Index{E: nm("sh"), Idx: ir.Binary{Op: "%", L: ir.Binary{Op: "+", L: lid, R: ir.Lit{Text: "1u"}}, R: ir.Lit{Text: "4u"}}}},
			},
		}},
	}
	out := make([]any, 4)
	if err := Run(m, "k", 4, &Memory{Vars: map[string]any{"out": out}}); err != nil {
		t.Fatal(err)
	}
	for i := range out {
		got := out[i].([]float64)
		if want := float64((i + 1) % 4); got[0] != want || got[1] != 1 || got[3] != 3 {
			t.Errorf("out[%d] = %v, want [%v 1 2 3]", i, got, want)
		}
	}
}
