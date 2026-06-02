package run

import (
	"testing"

	"m31labs.dev/elio/ir"
)

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// TestRunSkinLBS executes the linear-blend skinning kernel on the CPU fallback
// against a two-bone rig and three vertices, asserting the blended positions
// exactly. It proves the column-major palette layout transforms points with
// only vec×scalar / vec+vec — the ops Elio already has — and that weighted
// blending across influences accumulates correctly.
//
// Palette is column-major: bone j's four columns live at palette[j*4+0 .. j*4+3].
//   - bone0: identity
//   - bone1: identity + translate(10,0,0) (column 3 = {10,0,0,1})
func TestRunSkinLBS(t *testing.T) {
	mod := ir.SkinLBS()

	palette := []any{
		// bone0: identity (col0,col1,col2,col3)
		[]float64{1, 0, 0, 0},
		[]float64{0, 1, 0, 0},
		[]float64{0, 0, 1, 0},
		[]float64{0, 0, 0, 1},
		// bone1: identity + translate(10,0,0)
		[]float64{1, 0, 0, 0},
		[]float64{0, 1, 0, 0},
		[]float64{0, 0, 1, 0},
		[]float64{10, 0, 0, 1},
	}

	// Three vertices, all at rest position (1,2,3,1).
	restPos := []any{
		[]float64{1, 2, 3, 1},
		[]float64{1, 2, 3, 1},
		[]float64{1, 2, 3, 1},
	}
	joints := []any{
		[]float64{0, 0, 0, 0}, // v0: bone0 only
		[]float64{1, 0, 0, 0}, // v1: bone1 only
		[]float64{0, 1, 0, 0}, // v2: blend bone0 + bone1
	}
	weights := []any{
		[]float64{1, 0, 0, 0},
		[]float64{1, 0, 0, 0},
		[]float64{0.5, 0.5, 0, 0},
	}
	outPos := []any{nil, nil, nil}

	mem := &Memory{Vars: map[string]any{
		"restPos": restPos,
		"joints":  joints,
		"weights": weights,
		"palette": palette,
		"outPos":  outPos,
	}}

	if err := Run(mod, "main", len(restPos), mem); err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := [][3]float64{
		{1, 2, 3},  // v0: identity → unchanged
		{11, 2, 3}, // v1: translated +10 in x
		{6, 2, 3},  // v2: midpoint of (1,2,3) and (11,2,3)
	}
	for i, w := range want {
		got, ok := outPos[i].([]float64)
		if !ok || len(got) < 3 {
			t.Fatalf("outPos[%d] = %v, want a vec4", i, outPos[i])
		}
		for c := 0; c < 3; c++ {
			if abs(got[c]-w[c]) > 1e-9 {
				t.Errorf("outPos[%d] = (%g,%g,%g), want (%g,%g,%g)",
					i, got[0], got[1], got[2], w[0], w[1], w[2])
				break
			}
		}
	}
}
