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

// TestRunWhile executes a while loop summing 0..9 = 45, exercising the loop
// condition, compound assignment, and break-free termination.
func TestRunWhile(t *testing.T) {
	mod := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.U32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Var{Name: "i", Type: ir.U32, Init: ir.Lit{Text: "0u"}},
				ir.Var{Name: "sum", Type: ir.U32, Init: ir.Lit{Text: "0u"}},
				ir.While{
					Cond: ir.Binary{Op: "<", L: ir.Name{Name: "i"}, R: ir.Lit{Text: "10u"}},
					Body: []ir.Stmt{
						ir.Assign{Target: ir.Name{Name: "sum"}, Op: "+", Value: ir.Name{Name: "i"}},
						ir.Assign{Target: ir.Name{Name: "i"}, Op: "+", Value: ir.Lit{Text: "1u"}},
					},
				},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}}, Value: ir.Name{Name: "sum"}},
			},
		}},
	}
	acc := []float64{0}
	if err := Run(mod, "main", 1, &Memory{Vars: map[string]any{"acc": acc}}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if acc[0] != 45 {
		t.Errorf("acc[0] = %v, want 45 (0+1+…+9)", acc[0])
	}
}

// TestIntMultiplyI32Signed verifies that multiplying two negative i32 values
// yields the correct positive product (regression for the u32-only wrapping
// change that zero-extended the result and broke signed multiplies).
func TestIntMultiplyI32Signed(t *testing.T) {
	cases := []struct {
		a, b, want int64
		name       string
	}{
		{-3, -4, 12, "-3 * -4 = 12"},
		{-1, -1, 1, "-1 * -1 = 1"},
		{-100, 5, -500, "-100 * 5 = -500"},
		{7, -8, -56, "7 * -8 = -56"},
		// i32 min-value wrapping: -2147483648 * -1 overflows i32 → wraps to -2147483648
		{-2147483648, -1, -2147483648, "i32 min * -1 wraps"},
	}
	for _, c := range cases {
		got, err := intBinop("*", c.a, c.b)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", c.name, err)
			continue
		}
		if got.(int64) != c.want {
			t.Errorf("%s: got %d, want %d", c.name, got.(int64), c.want)
		}
	}
}

// TestIntMultiplyU32Wrapping verifies that u32 multiply wraps at 2^32 (low-32
// truncation), matching GPU u32 semantics used by the browser hash kernel.
func TestIntMultiplyU32Wrapping(t *testing.T) {
	// u32 values are represented as non-negative int64 in the interpreter.
	// The multiply must truncate to 32 bits. Low bits must match expected value.
	cases := []struct {
		a, b int64
		want uint32 // expected low-32 bit pattern
		name string
	}{
		// 0x45d9f3b is the browser hash multiplier; (2^32-1) * 0x45d9f3b mod 2^32
		// = (0 - 0x45d9f3b) mod 2^32 = 4294967296 - 73244475 = 4221722821 = 0xFBA260C5
		{0xFFFFFFFF, 0x45d9f3b, 0xFBA260C5, "u32_max * hash_mult"},
		// (2^32-1)^2 = 2^64 - 2*2^32 + 1; low32 = 1
		{0xFFFFFFFF, 0xFFFFFFFF, 1, "u32_max squared"},
		// 0x80000001 * 2 = 0x100000002; low32 = 2
		{0x80000001, 2, 2, "high_bit_set * 2"},
	}
	for _, c := range cases {
		got, err := intBinop("*", c.a, c.b)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", c.name, err)
			continue
		}
		// Compare only the low-32 bits — u32 result.
		gotLow := uint32(got.(int64))
		if gotLow != c.want {
			t.Errorf("%s: low32 = 0x%08X, want 0x%08X", c.name, gotLow, c.want)
		}
	}
}

// TestRunConst verifies a module-level constant is evaluated once and readable
// from a kernel: acc[i] = FACTOR (= 3).
func TestRunConst(t *testing.T) {
	mod := &ir.Module{
		Consts:   []ir.Const{{Name: "FACTOR", Type: ir.F32, Value: ir.Lit{Text: "3.0"}}},
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.F32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}}, Value: ir.Name{Name: "FACTOR"}},
			},
		}},
	}
	acc := []float64{0}
	if err := Run(mod, "main", 1, &Memory{Vars: map[string]any{"acc": acc}}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if acc[0] != 3 {
		t.Errorf("acc[0] = %v, want 3", acc[0])
	}
}
