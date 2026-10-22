package stdlib

import (
	"math"
	"math/rand"
	"testing"

	"m31labs.dev/elio/run"
)

// The foam and probe kernels must never form a large world coordinate in f32:
// the host passes window or point positions relative to a local origin and, per
// cascade, that origin wrapped into the tile (computed in float64). These tests
// run both kernels in the interpreter's float32 mode with the origin 1e6 m
// away and compare with a float64 reference evaluated at the absolute position.

func TestOceanFoamFarOriginFloat32(t *testing.T) {
	const w = 16
	var fb foamBuffers
	r := rand.New(rand.NewSource(7))
	n := [4]int{16, 8, 8, 4}
	tile := [4]float64{37, 5.3, 250, 1600}
	for c := 0; c < 4; c++ {
		fb[c] = make([][4]float64, n[c]*n[c])
		for i := range fb[c] {
			fb[c][i] = [4]float64{0, 0.5 * (r.Float64() - 0.5), 0.5 * (r.Float64() - 0.5), 0.3 * (r.Float64() - 0.5)}
		}
	}
	base := foamParams{width: w, cascades: 4, dt: 0.1, tau: 4, threshold: 0.95, gain: 3, lambda: 1, cell: 0.5, n: n, tile: tile}
	jac := func(p foamParams, wx, wz int) float64 {
		px, pz := (float64(wx)+0.5)*p.cell, (float64(wz)+0.5)*p.cell
		var sxx, szz, sxz float64
		for c := 0; c < 4; c++ {
			b := bilinear4(fb[c], p.n[c], p.tile[c], px, pz)
			sxx, szz, sxz = sxx+b[1], szz+b[2], sxz+b[3]
		}
		return (1+p.lambda*sxx)*(1+p.lambda*szz) - (p.lambda*sxz)*(p.lambda*sxz)
	}
	worst := func(p foamParams) (maxFoam, maxJ, spread float64) {
		state := constState(w, 0.4)
		got, tex := runOceanFoam(t, p, state, fb, run.Options{Float32: true})
		decay := math.Exp(-p.dt / p.tau)
		lo, hi := 1.0, 0.0
		for v := 0; v < w; v++ {
			for u := 0; u < w; u++ {
				wx, wz := p.originX+posMod(u-p.originX, w), p.originZ+posMod(v-p.originZ, w)
				j := jac(p, wx, wz)
				want := math.Min(0.4*decay+math.Min(math.Max((p.threshold-j)*p.gain, 0), 1), 1)
				maxFoam = math.Max(maxFoam, math.Abs(got[v*w+u]-want))
				maxJ = math.Max(maxJ, math.Abs(tex.Data[4*(v*w+u)+1]-j))
				lo, hi = math.Min(lo, want), math.Max(hi, want)
			}
		}
		return maxFoam, maxJ, hi - lo
	}
	near := base
	near.originX, near.originZ = -3, 11
	far := base
	far.originX, far.originZ = 2_000_003, -2_000_011 // 1e6 m at 0.5 m per cell
	nf, nj, ns := worst(near)
	ff, fj, fs := worst(far)
	t.Logf("near origin: foam error %.3g, J error %.3g (injection spread %.3f)", nf, nj, ns)
	t.Logf("far origin (1e6 m): foam error %.3g, J error %.3g (injection spread %.3f)", ff, fj, fs)
	if ns < 0.1 || fs < 0.1 {
		t.Errorf("injection barely varies (%g, %g): the test does not exercise the Jacobian", ns, fs)
	}
	if ff > 2*nf+1e-5 || fj > 2*nj+1e-5 {
		t.Errorf("far-origin error (foam %.3g, J %.3g) exceeds twice the near-origin error (%.3g, %.3g)", ff, fj, nf, nj)
	}
}

func TestOceanProbeFarOriginFloat32(t *testing.T) {
	const n, lambda, tm, step = 64, 0.6, 2.5, 0.25
	cs := oceanProbeCascades(t, n, tm, lambda)
	r := rand.New(rand.NewSource(11))
	const count = 32
	local := make([][2]float64, count)
	pts := make([]any, count)
	for i := range pts {
		local[i] = [2]float64{r.Float64()*100 - 50, r.Float64()*100 - 50}
		pts[i] = []float64{local[i][0], local[i][1], 0, 0}
	}
	run32 := func(ox, oz float64) (hErr, nErr float64) {
		var omX, omZ [4]float64
		for c := range cs {
			omX[c], omZ[c] = wrapTile(ox, cs[c].tile), wrapTile(oz, cs[c].tile)
		}
		out := runOceanProbeWith(t, cs, lambda, tm, step, pts, omX, omZ, run.Options{Float32: true})
		for i := range out {
			g := out[i].([]float64)
			want := probeRef(cs, ox+local[i][0], oz+local[i][1], step)
			hErr = math.Max(hErr, math.Abs(g[0]-want[0]))
			nErr = math.Max(nErr, math.Max(math.Abs(g[1]-want[1]), math.Abs(g[2]-want[2])))
		}
		return
	}
	nh, nn := run32(0, 0)
	fh, fn := run32(1_000_000.37, -1_000_000.81)
	t.Logf("near origin: height error %.3g m, normal error %.3g", nh, nn)
	t.Logf("far origin (1e6 m): height error %.3g m, normal error %.3g", fh, fn)
	if fh > 2*nh+1e-4 || fn > 2*nn+1e-4 {
		t.Errorf("far-origin error (%.3g m, %.3g) exceeds twice the near-origin error (%.3g m, %.3g)", fh, fn, nh, nn)
	}
	if nh > 1e-2 {
		t.Errorf("near-origin float32 height error %.3g m is too large to be a meaningful baseline", nh)
	}
}

