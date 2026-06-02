package run

import (
	"math"
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

// --- dual-quaternion helpers (in-test reference oracle) ---
//
// Quaternions are xyzw with w the scalar part. These mirror the kiln-core
// oracle the SkinDQS kernel is built to match: a Hamilton product, conjugate,
// dual-quat construction, blend (with antipodality), normalize, and point
// transform — all using the SAME formulas the kernel emits.

// qMul is the Hamilton product a*b (quaternions as xyzw).
func qMul(a, b [4]float64) [4]float64 {
	ax, ay, az, aw := a[0], a[1], a[2], a[3]
	bx, by, bz, bw := b[0], b[1], b[2], b[3]
	return [4]float64{
		aw*bx + ax*bw + ay*bz - az*by, // x
		aw*by - ax*bz + ay*bw + az*bx, // y
		aw*bz + ax*by - ay*bx + az*bw, // z
		aw*bw - ax*bx - ay*by - az*bz, // w
	}
}

// dualFromRotPos builds the dual part 0.5 * qMul((pos,0), rot) of a unit dual
// quaternion whose real part is the rotation rot and whose rigid translation is
// pos.
func dualFromRotPos(rot [4]float64, pos [3]float64) [4]float64 {
	t := qMul([4]float64{pos[0], pos[1], pos[2], 0}, rot)
	return [4]float64{0.5 * t[0], 0.5 * t[1], 0.5 * t[2], 0.5 * t[3]}
}

// dqsTransform reproduces the kernel's transform of point p under the blended,
// normalized dual quaternion (r, d) with blended uniform scale sb. It is the
// independent CPU oracle the kernel's CPU run is checked against.
func dqsTransform(r, d [4]float64, p, sb [3]float64) [3]float64 {
	// base = p * sb (componentwise)
	base := [3]float64{p[0] * sb[0], p[1] * sb[1], p[2] * sb[2]}

	// rotate base by r: uv = cross(r.xyz, base), uuv = cross(r.xyz, uv)
	rx, ry, rz, rw := r[0], r[1], r[2], r[3]
	uvx := ry*base[2] - rz*base[1]
	uvy := rz*base[0] - rx*base[2]
	uvz := rx*base[1] - ry*base[0]
	uuvx := ry*uvz - rz*uvy
	uuvy := rz*uvx - rx*uvz
	uuvz := rx*uvy - ry*uvx
	rotx := base[0] + 2*rw*uvx + 2*uuvx
	roty := base[1] + 2*rw*uvy + 2*uuvy
	rotz := base[2] + 2*rw*uvz + 2*uuvz

	// translation t = xyz of 2*qMul(d, conj(r))
	dx, dy, dz, dw := d[0], d[1], d[2], d[3]
	tx := 2 * (-dw*rx + dx*rw - dy*rz + dz*ry)
	ty := 2 * (-dw*ry + dx*rz + dy*rw - dz*rx)
	tz := 2 * (-dw*rz - dx*ry + dy*rx + dz*rw)

	return [3]float64{rotx + tx, roty + ty, rotz + tz}
}

// dqsReference blends the per-vertex influences (joints/weights) against the
// per-bone real/dual/scale palette exactly as the kernel does — antipodality
// flip against influence 0's real quat, accumulate, normalize by |accReal|,
// blend the uniform scale by Σw — then transforms p. This is the oracle.
func dqsReference(realQ, dualQ [][4]float64, boneScale []float64, j [4]int, w, p [4]float64) [3]float64 {
	ref := realQ[j[0]]
	var accReal, accDual [4]float64
	var scaleAcc, wsum float64
	for c := 0; c < 4; c++ {
		bk := j[c]
		rk := realQ[bk]
		dk := dualQ[bk]
		wk := w[c]
		wq := wk
		if c != 0 { // influence 0 seeds; dot(ref,ref) >= 0 so it never flips
			d := rk[0]*ref[0] + rk[1]*ref[1] + rk[2]*ref[2] + rk[3]*ref[3]
			if d < 0 {
				wq = -wk
			}
		}
		for k := 0; k < 4; k++ {
			accReal[k] += rk[k] * wq
			accDual[k] += dk[k] * wq
		}
		scaleAcc += boneScale[bk] * wk
		wsum += wk
	}
	inv := 1.0 / math.Sqrt(accReal[0]*accReal[0]+accReal[1]*accReal[1]+accReal[2]*accReal[2]+accReal[3]*accReal[3])
	var r, d [4]float64
	for k := 0; k < 4; k++ {
		r[k] = accReal[k] * inv
		d[k] = accDual[k] * inv
	}
	sb := scaleAcc * (1.0 / wsum)
	return dqsTransform(r, d, [3]float64{p[0], p[1], p[2]}, [3]float64{sb, sb, sb})
}

// TestRunSkinDQS is the correctness anchor for the dual-quaternion skinning
// kernel: it runs SkinDQS on the CPU fallback over a two-bone rig and asserts
// the skinned positions against an independent in-test DQS oracle AND against
// hand-derived values. The palette (real/dual quats per bone) is built in the
// test from rotation+translation, exactly as the host does.
//
// Rig:
//   - bone0: 90° about +Z (sends (1,0,0)->(0,1,0)), translate (5,0,0)
//   - bone1: identity rotation, translate (0,3,0)
//   - bone2: -rot0 — the SAME 90°-about-+Z rotation as bone0 but its quaternion
//     lives in the opposite hemisphere (negated). dot(rot2, rot0) = -1 < 0, so
//     any influence on bone2 blended against a bone0 reference must trigger the
//     kernel's antipodality flip (wq = -w). Translate (0,0,7).
//   - bone3: 120° about the normalized (1,1,1) axis — quaternion (.5,.5,.5,.5).
//     This is an OFF-AXIS rotation: applied to a non-axis-aligned point it makes
//     the cross-product terms uvx/uvz/uuvx/uuvz and the dual-w translation terms
//     tx/ty/tz all non-zero (they are identically 0 for the +Z-only bones above).
//     It cycles axes (x,y,z)->(z,x,y). Translate (1,2,3).
//   - uniform scale 1 on all.
//
// Vertices:
//   - v0: 100% bone0 at p=(1,0,0) -> rotate to (0,1,0) + (5,0,0) = (5,1,0)
//   - v1: 100% bone1 at p=(2,2,2) -> identity + (0,3,0) = (2,5,2)
//   - v2: 50/50 bone0+bone1 at p=(1,0,0)
//   - v3: 50/50 bone0+bone2 at p=(1,2,3) — antipodal-flip path. influence 0 is
//     bone0 (the reference); influence 1 is bone2 = -rot0, so dot < 0 and the
//     flip fires. After the flip both influences describe the same rotation, so
//     the real part blends back to rot0; only the translation (pos0 vs pos2)
//     blends. Expected comes from the in-test oracle (which performs the same
//     flip).
//   - v4: 100% bone3 at p=(1,2,3) — off-axis rotation. (x,y,z)->(z,x,y) gives
//     (3,1,2), + pos3 (1,2,3) = (4,3,5). Exercises the cross/translation terms
//     the +Z-only rig leaves at zero.
func TestRunSkinDQS(t *testing.T) {
	mod := ir.SkinDQS()

	s := math.Sqrt2 / 2                    // sin(pi/4) = cos(pi/4)
	rot0 := [4]float64{0, 0, s, s}         // 90 deg about +Z
	rot1 := [4]float64{0, 0, 0, 1}         // identity
	rot2 := [4]float64{0, 0, -s, -s}       // -rot0: same rotation, opposite hemisphere
	rot3 := [4]float64{0.5, 0.5, 0.5, 0.5} // 120 deg about normalized (1,1,1)
	pos0 := [3]float64{5, 0, 0}
	pos1 := [3]float64{0, 3, 0}
	pos2 := [3]float64{0, 0, 7}
	pos3 := [3]float64{1, 2, 3}

	realBones := [][4]float64{rot0, rot1, rot2, rot3}
	dualBones := [][4]float64{
		dualFromRotPos(rot0, pos0),
		dualFromRotPos(rot1, pos1),
		dualFromRotPos(rot2, pos2),
		dualFromRotPos(rot3, pos3),
	}
	scaleBones := []float64{1, 1, 1, 1}

	// Buffers for the kernel (as []any of []float64), indexed directly by bone.
	realQ := make([]any, len(realBones))
	dualQ := make([]any, len(dualBones))
	boneScale := make([]any, len(realBones))
	for b := range realBones {
		realQ[b] = []float64{realBones[b][0], realBones[b][1], realBones[b][2], realBones[b][3]}
		dualQ[b] = []float64{dualBones[b][0], dualBones[b][1], dualBones[b][2], dualBones[b][3]}
		boneScale[b] = []float64{scaleBones[b], scaleBones[b], scaleBones[b], 0}
	}

	type vert struct {
		p [4]float64
		j [4]int
		w [4]float64
	}
	verts := []vert{
		{p: [4]float64{1, 0, 0, 1}, j: [4]int{0, 0, 0, 0}, w: [4]float64{1, 0, 0, 0}},
		{p: [4]float64{2, 2, 2, 1}, j: [4]int{1, 0, 0, 0}, w: [4]float64{1, 0, 0, 0}},
		{p: [4]float64{1, 0, 0, 1}, j: [4]int{0, 1, 0, 0}, w: [4]float64{0.5, 0.5, 0, 0}},
		// v3: antipodal-flip path (influence 1 = bone2 = -rot0, dot < 0).
		{p: [4]float64{1, 2, 3, 1}, j: [4]int{0, 2, 0, 0}, w: [4]float64{0.5, 0.5, 0, 0}},
		// v4: off-axis rotation (bone3, 120° about (1,1,1)) at a non-axis point.
		{p: [4]float64{1, 2, 3, 1}, j: [4]int{3, 0, 0, 0}, w: [4]float64{1, 0, 0, 0}},
	}

	restPos := make([]any, len(verts))
	joints := make([]any, len(verts))
	weights := make([]any, len(verts))
	dst := make([]any, len(verts))
	for i, v := range verts {
		restPos[i] = []float64{v.p[0], v.p[1], v.p[2], v.p[3]}
		joints[i] = []float64{float64(v.j[0]), float64(v.j[1]), float64(v.j[2]), float64(v.j[3])}
		weights[i] = []float64{v.w[0], v.w[1], v.w[2], v.w[3]}
		// dst is written componentwise (dst[i].x = ...), so each cell must be a
		// pre-allocated vec4 — as the host allocates the output buffer.
		dst[i] = []float64{0, 0, 0, 0}
	}

	mem := &Memory{Vars: map[string]any{
		"restPos":   restPos,
		"joints":    joints,
		"weights":   weights,
		"realQ":     realQ,
		"dualQ":     dualQ,
		"boneScale": boneScale,
		"dst":       dst,
	}}

	if err := Run(mod, "main", len(verts), mem); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Hand-derived spot checks. v3's real part blends back to bone0's rotation
	// (bone2 is the same rotation in the opposite hemisphere, un-flipped by the
	// kernel), so it rotates (1,2,3)->(-2,1,3) and the translation is the 50/50
	// blend of pos0(5,0,0) and pos2(0,0,7) = (2.5,0,3.5): (0.5,1,6.5).
	hand := map[int][3]float64{
		0: {5, 1, 0},     // 90deg-Z: (1,0,0)->(0,1,0), + pos0(5,0,0)
		1: {2, 5, 2},     // identity, + pos1(0,3,0)
		3: {0.5, 1, 6.5}, // antipodal-flip: (1,2,3)->(-2,1,3), + (2.5,0,3.5)
		4: {4, 3, 5},     // off-axis 120° (1,1,1): (1,2,3)->(3,1,2), + pos3(1,2,3)
	}

	for i, v := range verts {
		want := dqsReference(realBones, dualBones, scaleBones, v.j, v.w, v.p)
		if h, ok := hand[i]; ok {
			for c := 0; c < 3; c++ {
				if abs(want[c]-h[c]) > 1e-6 {
					t.Fatalf("oracle disagrees with hand-derived for v%d: oracle=%v hand=%v", i, want, h)
				}
			}
		}
		got, ok := dst[i].([]float64)
		if !ok || len(got) < 4 {
			t.Fatalf("dst[%d] = %v, want a vec4", i, dst[i])
		}
		for c := 0; c < 3; c++ {
			// NaN/Inf must fail: a bare `abs(d) > tol` comparison is false for
			// NaN, so a NaN/Inf-producing defect would slip through silently.
			d := got[c] - want[c]
			if math.IsNaN(d) || math.IsInf(d, 0) || math.Abs(d) > 1e-6 {
				t.Fatalf("dst[%d] = (%g,%g,%g), want (%g,%g,%g)",
					i, got[0], got[1], got[2], want[0], want[1], want[2])
			}
		}
		dw := got[3] - 1
		if math.IsNaN(dw) || math.IsInf(dw, 0) || math.Abs(dw) > 1e-9 {
			t.Errorf("dst[%d].w = %g, want 1", i, got[3])
		}
	}
}

// TestRunMemberAssign exercises the interpreter's member-assign support — the
// one new statement shape the dual-quaternion kernel needs to write
// dst[i].x = ... componentwise. It builds an inline module with a single
// read_write vec4 buffer and a kernel that writes 7 into dst[i].x, then asserts
// the slice cell was mutated in place.
func TestRunMemberAssign(t *testing.T) {
	mod := &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "dst", Type: ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}}},
		},
		Kernels: []ir.Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
				ir.Assign{
					Target: ir.Member{E: ir.Index{E: ir.Name{Name: "dst"}, Idx: ir.Name{Name: "i"}}, Field: "x"},
					Value:  ir.Lit{Text: "7.0"},
				},
			},
		}},
	}

	dst := []any{[]float64{0, 0, 0, 0}}
	mem := &Memory{Vars: map[string]any{"dst": dst}}
	if err := Run(mod, "main", 1, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := dst[0].([]float64)
	if abs(got[0]-7) > 1e-9 {
		t.Fatalf("dst[0].x = %v, want 7", got)
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
