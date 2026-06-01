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
