package run

import (
	"testing"

	"m31labs.dev/elio/ir"
)

// translate builds a column-major mat4 translation (basis = identity,
// column 3 = (tx,ty,tz,1)).
func translate(tx, ty, tz float64) Mat {
	return Mat{Cols: 4, Rows: 4, E: []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		tx, ty, tz, 1,
	}}
}

func record(tx, ty, tz float64) map[string]any {
	return map[string]any{
		"model":    translate(tx, ty, tz),
		"pickData": []float64{0, 0, 0, 0},
	}
}

func translationOf(rec any) [3]float64 {
	m := rec.(map[string]any)["model"].(Mat)
	return [3]float64{m.E[12], m.E[13], m.E[14]}
}

// TestRunCull executes the Elio cull kernel on the CPU fallback against three
// instances and an axis-aligned box frustum [-10,10]^3, and asserts the correct
// instances survive in the correct compacted order. This is the first real
// *execution* of an Elio kernel (the WGSL path is naga-validated separately).
func TestRunCull(t *testing.T) {
	mod := ir.CullKernel()

	// Box frustum [-10,10]^3: inside iff dot(n, p) + d >= 0 for all planes.
	planes := []any{
		[]float64{1, 0, 0, 10},  // x >= -10
		[]float64{-1, 0, 0, 10}, // x <= 10
		[]float64{0, 1, 0, 10},  // y >= -10
		[]float64{0, -1, 0, 10}, // y <= 10
		[]float64{0, 0, 1, 10},  // z >= -10
		[]float64{0, 0, -1, 10}, // z <= 10
	}

	in := []any{
		record(0, 0, 0),   // inside
		record(100, 0, 0), // outside (x = 100)
		record(5, -5, 5),  // inside
	}
	out := []any{nil, nil, nil}
	drawArgs := []float64{36, 0, 0, 0} // [vertexCount, instanceCount, firstVertex, firstInstance]

	mem := &Memory{Vars: map[string]any{
		"cull": map[string]any{
			"planes":      planes,
			"vertexCount": float64(36),
			"radius":      float64(0),
		},
		"input":    in,
		"output":   out,
		"drawArgs": drawArgs,
	}}

	if err := Run(mod, "main", len(in), mem); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got := int(drawArgs[1]); got != 2 {
		t.Fatalf("instanceCount = %d, want 2", got)
	}
	if got := translationOf(out[0]); got != [3]float64{0, 0, 0} {
		t.Errorf("survivor 0 translation = %v, want {0 0 0}", got)
	}
	if got := translationOf(out[1]); got != [3]float64{5, -5, 5} {
		t.Errorf("survivor 1 translation = %v, want {5 -5 5}", got)
	}
	// vertexCount must be preserved for the indirect draw.
	if drawArgs[0] != 36 {
		t.Errorf("vertexCount = %v, want 36", drawArgs[0])
	}
}

// TestRunReduce executes the workgroup tree-reduction in lockstep across two
// 64-wide workgroups over src[i] = i, and asserts each workgroup's partial sum.
// This is the proof that the CPU fallback now runs cooperating workgroups
// (shared memory + barriers), not only embarrassingly-parallel kernels — and,
// under -race, that the kernel's shared access is correctly synchronized.
func TestRunReduce(t *testing.T) {
	mod := ir.WorkgroupReduce()
	const n = 128 // two workgroups of 64
	src := make([]float64, n)
	for i := range src {
		src[i] = float64(i)
	}
	partials := make([]float64, n/64)
	mem := &Memory{Vars: map[string]any{"src": src, "partials": partials}}

	if err := Run(mod, "reduce", n, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	// workgroup 0 sums 0..63 = 2016; workgroup 1 sums 64..127 = 6112.
	if partials[0] != 2016 {
		t.Errorf("partials[0] = %v, want 2016", partials[0])
	}
	if partials[1] != 6112 {
		t.Errorf("partials[1] = %v, want 6112", partials[1])
	}
}

// TestRunCompoundAssign verifies compound assignment evaluates as target = target
// Op value: (10 + 5) * 2 = 30.
func TestRunCompoundAssign(t *testing.T) {
	mod := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.F32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Var{Name: "x", Init: ir.Lit{Text: "10.0"}},
				ir.Assign{Target: ir.Name{Name: "x"}, Op: "+", Value: ir.Lit{Text: "5.0"}},
				ir.Assign{Target: ir.Name{Name: "x"}, Op: "*", Value: ir.Lit{Text: "2.0"}},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}}, Value: ir.Name{Name: "x"}},
			},
		}},
	}
	acc := []float64{0}
	if err := Run(mod, "main", 1, &Memory{Vars: map[string]any{"acc": acc}}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if acc[0] != 30 {
		t.Errorf("acc[0] = %v, want 30", acc[0])
	}
}
