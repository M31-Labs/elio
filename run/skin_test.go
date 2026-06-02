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

// TestRunSkinLBSRotation exercises the kernel's headline feature — column-major
// handling of the ROTATION columns (col0/col1/col2) — which the translation-only
// rig in TestRunSkinLBS never touches. The bone is a 90° rotation about +Z stored
// column-major:
//
//	col0 = {0,1,0,0}  col1 = {-1,0,0,0}  col2 = {0,0,1,0}  col3 = {0,0,0,1}
//
// For rest vertex p = (1,2,3,1) at full weight, M·p is
//
//	col0*p.x + col1*p.y + col2*p.z + col3*p.w
//	= {0,1,0,0}*1 + {-1,0,0,0}*2 + {0,0,1,0}*3 + {0,0,0,1}*1
//	= {-2, 1, 3, 1}
//
// i.e. (x,y,z) -> (-p.y, p.x, p.z) = (-2, 1, 3). This pins down that transform
// maps p.x->col0, p.y->col1, p.z->col2: a transposed or dropped column would
// fail here even though the translation-only test still passes.
func TestRunSkinLBSRotation(t *testing.T) {
	mod := ir.SkinLBS()

	palette := []any{
		// bone0: 90° rotation about +Z (column-major)
		[]float64{0, 1, 0, 0},  // col0
		[]float64{-1, 0, 0, 0}, // col1
		[]float64{0, 0, 1, 0},  // col2
		[]float64{0, 0, 0, 1},  // col3
	}

	restPos := []any{
		[]float64{1, 2, 3, 1},
	}
	joints := []any{
		[]float64{0, 0, 0, 0}, // v0: bone0 only
	}
	weights := []any{
		[]float64{1, 0, 0, 0}, // full weight on bone0
	}
	outPos := []any{nil}

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

	want := [3]float64{-2, 1, 3} // x = -p.y, y = p.x, z = p.z
	got, ok := outPos[0].([]float64)
	if !ok || len(got) < 3 {
		t.Fatalf("outPos[0] = %v, want a vec4", outPos[0])
	}
	for c := 0; c < 3; c++ {
		if abs(got[c]-want[c]) > 1e-9 {
			t.Fatalf("outPos[0] = (%g,%g,%g), want (%g,%g,%g)",
				got[0], got[1], got[2], want[0], want[1], want[2])
		}
	}
}

// TestRunSqrt exercises the sqrt builtin via SqrtKernel: out = p * |p|.
func TestRunSqrt(t *testing.T) {
	a := []any{[]float64{3, 4, 0, 0}, []float64{0, 0, 0, 0}}
	out := []any{nil, nil}
	mem := &Memory{Vars: map[string]any{"a": a, "dst": out}}
	if err := Run(ir.SqrtKernel(), "main", 2, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	g := out[0].([]float64) // |{3,4,0,0}| = 5 → {15,20,0,0}
	if abs(g[0]-15) > 1e-9 || abs(g[1]-20) > 1e-9 || abs(g[2]) > 1e-9 {
		t.Fatalf("sqrt kernel out[0] = %v, want {15,20,0,0}", g)
	}
}
