package stdlib

import (
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func TestSkinIsValid(t *testing.T) {
	if errs := sema.Check(Skin()); len(errs) != 0 {
		t.Fatalf("Skin failed sema:\n%v", sema.Errors(errs))
	}
}

// TestSkinBlendsBones skins one vertex at (1,2,3) by two bones — an identity and
// a +10x translation — weighted 50/50, and checks the blended result lands at
// the midpoint (6,2,3). This exercises matrix-column-expanded transforms, the
// weighted blend, and struct + u32-index reads on the CPU fallback.
func TestSkinBlendsBones(t *testing.T) {
	identity := run.Mat{Cols: 4, Rows: 4, E: []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}}
	translateX10 := run.Mat{Cols: 4, Rows: 4, E: []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		10, 0, 0, 1, // column 3 = translation
	}}
	bones := []any{identity, translateX10}

	vert := map[string]any{
		"px": 1.0, "py": 2.0, "pz": 3.0,
		"w0": 0.5, "w1": 0.5, "w2": 0.0, "w3": 0.0,
		"b0": int64(0), "b1": int64(1), "b2": int64(0), "b3": int64(0),
	}
	out := []float64{0, 0, 0}
	mem := &run.Memory{Vars: map[string]any{"bones": bones, "verts": []any{vert}, "out": out}}

	if err := run.Run(Skin(), "skin", 1, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	// 0.5*(1,2,3) + 0.5*(11,2,3) = (6,2,3)
	want := []float64{6, 2, 3}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out = %v, want %v (mismatch at [%d])", out, want, i)
		}
	}
}
