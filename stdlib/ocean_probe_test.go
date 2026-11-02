package stdlib

import (
	"math"
	"math/rand"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// probeCascade is one cascade. raw is the post-FFT buffer A the probe reads:
// n*n vec4 in natural order, (Dy, Dx, Dz, dDy/dx), NOT scaled by choppiness.
// d is the reference's view of it, (Dx, Dy, Dz, dDy/dx) with lambda applied to
// Dx and Dz, which the kernel must reproduce from raw and the uniform lambda.
type probeCascade struct {
	n    int
	tile float64
	raw  [][4]float64
	d    [][4]float64
}

// bilinear returns the displacement (Dx, Dy, Dz) and the derivatives of Dx and
// Dz with respect to world x and z at (px, pz): the piecewise-bilinear field and
// its exact cell gradient, as the kernel computes them.
func (c probeCascade) bilinear(px, pz float64) (d [3]float64, dx, dz [3]float64) {
	// Texel (i, j) sits at world offset ((i+0.5), (j+0.5)) * tile / n, as the
	// hardware sampler places it: the sample position in texels is u*n - 0.5.
	u, v := px/c.tile, pz/c.tile
	fx, fz := (u-math.Floor(u))*float64(c.n)-0.5, (v-math.Floor(v))*float64(c.n)-0.5
	ix, iz := math.Floor(fx), math.Floor(fz)
	rx, rz := fx-ix, fz-iz
	x0, z0 := (int(ix)+c.n)%c.n, (int(iz)+c.n)%c.n
	x1, z1 := (x0+1)%c.n, (z0+1)%c.n
	h := c.tile / float64(c.n)
	for k := 0; k < 3; k++ {
		v00, v10 := c.d[z0*c.n+x0][k], c.d[z0*c.n+x1][k]
		v01, v11 := c.d[z1*c.n+x0][k], c.d[z1*c.n+x1][k]
		d[k] = (1-rz)*((1-rx)*v00+rx*v10) + rz*((1-rx)*v01+rx*v11)
		dx[k] = ((1-rz)*(v10-v00) + rz*(v11-v01)) / h
		dz[k] = ((1-rx)*(v01-v00) + rx*(v11-v10)) / h
	}
	return
}

// sampleRepeat emulates textureSample on an n x n texture with a repeat
// address mode, a linear filter and level zero (WebGPU spec: texel centres at
// (i+0.5)/n, the sample position is u*n-0.5, each of the four texels wraps by
// its own integer index). It is written apart from bilinear, from the
// unwrapped coordinate, so the two cross-check each other. It returns the
// first three lanes (Dx, Dy, Dz) of the lambda-scaled field.
func (c probeCascade) sampleRepeat(u, v float64) [3]float64 {
	wrap := func(i int) int { return ((i % c.n) + c.n) % c.n }
	x, z := u*float64(c.n)-0.5, v*float64(c.n)-0.5
	i0, j0 := int(math.Floor(x)), int(math.Floor(z))
	wx, wz := x-float64(i0), z-float64(j0)
	var r [3]float64
	for k := 0; k < 3; k++ {
		t := func(i, j int) float64 { return c.d[wrap(j)*c.n+wrap(i)][k] }
		r[k] = (1-wz)*((1-wx)*t(i0, j0)+wx*t(i0+1, j0)) + wz*((1-wx)*t(i0, j0+1)+wx*t(i0+1, j0+1))
	}
	return r
}

// probeSum sums every cascade at (px, pz).
func probeSum(cs []probeCascade, px, pz float64) (d [3]float64, dx, dz [3]float64) {
	for _, c := range cs {
		a, b, e := c.bilinear(px, pz)
		for k := 0; k < 3; k++ {
			d[k] += a[k]
			dx[k] += b[k]
			dz[k] += e[k]
		}
	}
	return
}

// probeHeightRef inverts the horizontal displacement at world point (qx, qz)
// with fixedSteps fixed-point steps then newtonSteps Newton steps, and returns
// the height there.
func probeHeightRef(cs []probeCascade, qx, qz float64, fixedSteps, newtonSteps int) float64 {
	px, pz := probeInvertRef(cs, qx, qz, fixedSteps, newtonSteps)
	d, _, _ := probeSum(cs, px, pz)
	return d[1]
}

// probeInvertRef returns the rest position p with p + D_h(p) close to q.
func probeInvertRef(cs []probeCascade, qx, qz float64, fixedSteps, newtonSteps int) (float64, float64) {
	px, pz := qx, qz
	for it := 0; it < fixedSteps+newtonSteps; it++ {
		d, dx, dz := probeSum(cs, px, pz)
		if it < fixedSteps {
			px, pz = qx-d[0], qz-d[2]
			continue
		}
		fx, fz := px+d[0]-qx, pz+d[2]-qz
		a, b, c, e := 1+dx[0], dz[0], dx[2], 1+dz[2]
		det := a*e - b*c
		if math.Abs(det) < 1e-6 {
			continue
		}
		px -= (e*fx - b*fz) / det
		pz -= (a*fz - c*fx) / det
	}
	return px, pz
}

// probeRef returns (height, normal.x, normal.z) as the kernel computes them.
func probeRef(cs []probeCascade, qx, qz, step float64) [3]float64 {
	h := func(x, z float64) float64 { return probeHeightRef(cs, x, z, 2, 4) }
	hx := (h(qx+step, qz) - h(qx-step, qz)) / (2 * step)
	hz := (h(qx, qz+step) - h(qx, qz-step)) / (2 * step)
	s := math.Sqrt(hx*hx + hz*hz + 1)
	return [3]float64{h(qx, qz), -hx / s, -hz / s}
}

// oceanProbeCascades builds four cascade displacement buffers through the real
// pipeline (spectrum, evolve, inverse FFT) at time tm with choppiness lambda.
func oceanProbeCascades(t *testing.T, n int, tm, lambda float64) []probeCascade {
	t.Helper()
	var cs []probeCascade
	for c := 0; c < 4; c++ {
		s := defaultSpec(c, n)
		s.seed = 3
		h0, omega := runOceanSpectrum(t, s)
		f := runOceanEvolve(t, s, h0, omega, tm)
		pc := probeCascade{n: n, tile: s.tile, raw: make([][4]float64, n*n), d: make([][4]float64, n*n)}
		for i := range pc.d {
			pc.raw[i] = [4]float64{f[0][i], f[1][i], f[2][i], f[3][i]}
			pc.d[i] = [4]float64{lambda * f[1][i], f[0][i], lambda * f[2][i], f[3][i]}
		}
		cs = append(cs, pc)
	}
	return cs
}

// runOceanProbe runs the kernel over pts with the cascades' raw buffers and the
// choppiness lambda in the uniform, and returns the out buffer.
func runOceanProbe(t *testing.T, cs []probeCascade, lambda, tm, step float64, pts []any) []any {
	return runOceanProbeWith(t, cs, lambda, tm, step, pts, [4]float64{}, [4]float64{}, run.Options{})
}

// runOceanProbeWith also takes the per-cascade wrapped origin offsets (the
// points are relative to an origin whose offset in cascade c's tile is
// omX[c], omZ[c]) and interpreter options.
func runOceanProbeWith(t *testing.T, cs []probeCascade, lambda, tm, step float64, pts []any, omX, omZ [4]float64, opts run.Options) []any {
	t.Helper()
	m, err := OceanProbe()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	count := len(pts)
	bufs := make([]any, 4)
	for c := range bufs {
		n := cs[c].n
		b := make([]any, n*n)
		for i := range b {
			b[i] = []float64{cs[c].raw[i][0], cs[c].raw[i][1], cs[c].raw[i][2], cs[c].raw[i][3]}
		}
		bufs[c] = b
	}
	out := make([]any, count)
	for i := range out {
		out[i] = []float64{9, 9, 9, 9}
	}
	u := map[string]any{"count": int64(count), "cascades": int64(4), "_pad0": int64(0), "_pad1": int64(0),
		"t": tm, "normalStep": step, "lambda": lambda, "_pad3": 0.0}
	for c := 0; c < 4; c++ {
		u[probeField("n", c)] = int64(cs[c].n)
		u[probeField("size", c)] = cs[c].tile
		u[probeField("omX", c)] = omX[c]
		u[probeField("omZ", c)] = omZ[c]
	}
	mem := &run.Memory{Vars: map[string]any{
		"probe": u, "points": pts, "disp0": bufs[0], "disp1": bufs[1], "disp2": bufs[2], "disp3": bufs[3], "out": out,
	}}
	if err := run.RunWith(m, "ocean_probe", count, mem, opts); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestOceanProbeMatchesReference(t *testing.T) {
	const n, lambda, tm, step = 64, 0.6, 2.5, 0.25
	cs := oceanProbeCascades(t, n, tm, lambda)
	r := rand.New(rand.NewSource(42))
	const count = 64
	pts := make([]any, count)
	for i := range pts {
		pts[i] = []float64{r.Float64()*4000 - 2000, r.Float64()*4000 - 2000, 0, 0}
	}
	out := runOceanProbe(t, cs, lambda, tm, step, pts)
	var worstH, worstN, worstConv, maxShift float64
	var converged int
	for i := 0; i < count; i++ {
		g := out[i].([]float64)
		q := pts[i].([]float64)
		want := probeRef(cs, q[0], q[1], step)
		eh := math.Abs(g[0] - want[0])
		en := math.Max(math.Abs(g[1]-want[1]), math.Abs(g[2]-want[2]))
		if eh > worstH {
			worstH = eh
		}
		if en > worstN {
			worstN = en
		}
		if g[3] != tm {
			t.Errorf("probe %d: time lane %g, want %g", i, g[3], tm)
		}
		if eh > 1e-6 {
			t.Errorf("probe %d at (%.1f, %.1f): height %.9g, reference %.9g (error %.3g m)", i, q[0], q[1], g[0], want[0], eh)
		}
		if en > 1e-6 {
			t.Errorf("probe %d: normal (%g, %g), reference (%g, %g)", i, g[1], g[2], want[1], want[2])
		}
		// Compare with a fully converged inversion where the surface has not
		// folded: this is the truncation error of two fixed-point and four Newton steps.
		conv := probeHeightRef(cs, q[0], q[1], 2, 40)
		more := probeHeightRef(cs, q[0], q[1], 2, 60)
		if math.Abs(conv-more) < 1e-9 {
			converged++
			// The converged rest position lands on the query.
			px, pz := probeInvertRef(cs, q[0], q[1], 2, 40)
			d, _, _ := probeSum(cs, px, pz)
			if r := math.Hypot(px+d[0]-q[0], pz+d[2]-q[1]); r > 1e-9 {
				t.Errorf("probe %d: inverted point misses the query by %g m", i, r)
			}
			// The horizontal displacement matters: ignoring it changes the height.
			plain, _, _ := probeSum(cs, q[0], q[1])
			if dd := math.Abs(plain[1] - conv); dd > maxShift {
				maxShift = dd
			}
			if e := math.Abs(g[0] - conv); e > worstConv {
				worstConv = e
			}
		}
	}
	t.Logf("probe vs CPU evaluation of the same field (%d points): max height error %.3g m, max normal error %.3g", count, worstH, worstN)
	t.Logf("probe vs converged inversion (%d of %d points converged): max height difference %.3g m", converged, count, worstConv)
	t.Logf("largest height change from inverting the displacement: %.3g m", maxShift)
	if maxShift < 0.01 {
		t.Errorf("inversion changes no height by more than 1 cm (%.3g m): the test does not exercise it", maxShift)
	}
	if converged < count/2 {
		t.Errorf("only %d of %d points converged; the test field is too choppy", converged, count)
	}
	if worstConv > 1e-3 {
		t.Errorf("six-step inversion differs from the converged one by %.3g m", worstConv)
	}
}

func probeField(prefix string, c int) string { return prefix + string(rune('0'+c)) }

// TestOceanProbeAppliesLambda checks that the kernel scales Dx and Dz by the
// uniform lambda, not the host: at lambda 0 the probe reads the height at the
// query point itself (no inversion), and the same raw buffers at lambda 0.6 give
// the reference answer for lambda 0.6.
func TestOceanProbeAppliesLambda(t *testing.T) {
	const n, tm, step = 64, 2.5, 0.25
	cs := oceanProbeCascades(t, n, tm, 0) // d holds lambda 0
	pts := []any{[]float64{123.4, -567.8, 0, 0}, []float64{-1500.25, 800.5, 0, 0}}
	out := runOceanProbe(t, cs, 0, tm, step, pts)
	for i, p := range pts {
		q := p.([]float64)
		plain, _, _ := probeSum(cs, q[0], q[1])
		if got := out[i].([]float64)[0]; math.Abs(got-plain[1]) > 1e-6 {
			t.Errorf("lambda 0, point %d: height %.9g, want the undisplaced sum of Dy %.9g", i, got, plain[1])
		}
	}
	// The same raw buffers with lambda 0.6 match a reference built with 0.6.
	const lambda = 0.6
	for _, c := range cs {
		for i := range c.d {
			c.d[i][0], c.d[i][2] = lambda*c.raw[i][1], lambda*c.raw[i][2]
		}
	}
	out = runOceanProbe(t, cs, lambda, tm, step, pts)
	for i, p := range pts {
		q := p.([]float64)
		want := probeRef(cs, q[0], q[1], step)
		g := out[i].([]float64)
		if math.Abs(g[0]-want[0]) > 1e-6 || math.Abs(g[1]-want[1]) > 1e-6 || math.Abs(g[2]-want[2]) > 1e-6 {
			t.Errorf("lambda %g, point %d: got (%g, %g, %g), want %v", lambda, i, g[0], g[1], g[2], want)
		}
	}
}

// samplerSurface returns the surface the ocean shader draws for the rest
// position (px, pz), both absolute world metres: the cascades sampled like the
// hardware sampler, summed, the horizontal displacement added to the position.
// It returns the drawn world position (x, z), the height, and the Jacobian
// determinant of the horizontal map (1 + dDx/dx)(1 + dDz/dz) - (dDx/dz)(dDz/dx).
func samplerSurface(cs []probeCascade, px, pz float64) (qx, qz, h, det float64) {
	var d [3]float64
	for _, c := range cs {
		s := c.sampleRepeat(px/c.tile, pz/c.tile)
		for k := range d {
			d[k] += s[k]
		}
	}
	_, dx, dz := probeSum(cs, px, pz)
	det = (1 + dx[0]) * (1 + dz[2]) - dz[0]*dx[2]
	return px + d[0], pz + d[2], d[1], det
}

// TestProbeReferenceSamplesLikeHardware checks that the reference's bilinear
// fetch is the sampler's, including at the tile seam and at texel centres.
func TestProbeReferenceSamplesLikeHardware(t *testing.T) {
	cs := oceanProbeCascades(t, 16, 1.5, 0.6)
	r := rand.New(rand.NewSource(7))
	pos := []float64{0, 1, -1, 0.5 / 16, 1 - 0.5/16, 1.0 / 16, 0.999999, -0.000001}
	for i := 0; i < 200; i++ {
		pos = append(pos, r.Float64()*6-3)
	}
	for _, c := range cs {
		for _, u := range pos {
			for _, v := range pos[:12] {
				got, _, _ := c.bilinear(u*c.tile, v*c.tile)
				want := c.sampleRepeat(u, v)
				for k := 0; k < 3; k++ {
					if math.Abs(got[k]-want[k]) > 1e-9 {
						t.Fatalf("tile %g at (%g, %g) lane %d: bilinear %g, sampler %g", c.tile, u, v, k, got[k], want[k])
					}
				}
			}
		}
		// A texel centre returns the texel itself.
		i, j := 5, 11
		got := c.sampleRepeat((float64(i)+0.5)/float64(c.n), (float64(j)+0.5)/float64(c.n))
		if got[1] != c.d[j*c.n+i][1] {
			t.Errorf("texel centre (%d, %d): %g, want %g", i, j, got[1], c.d[j*c.n+i][1])
		}
	}
}

// TestOceanProbeMatchesDrawnSurface compares the kernel with the surface the
// shader draws. For rest positions P it forms the drawn point q = P + D_h(P)
// and height Dy(P) with a sampler emulation (texel centres, repeat, bilinear,
// every cascade, the same lambda), then asks the kernel for the height at q. The
// probe points are relative to an origin O, with the per-cascade offsets
// mod(O, tile) as the host passes them; the cases cover O = 0, a far origin, and
// rest positions on and next to tile seams and texel centres.
func TestOceanProbeMatchesDrawnSurface(t *testing.T) {
	const n, lambda, tm, step = 64, 0.6, 2.5, 0.25
	cs := oceanProbeCascades(t, n, tm, lambda)
	origins := [][2]float64{{0, 0}, {1234.5, -987.25}, {-40001.75, 30000.5}}
	r := rand.New(rand.NewSource(11))
	var worstH float64
	for _, o := range origins {
		var rest [][2]float64
		for i := 0; i < 48; i++ {
			rest = append(rest, [2]float64{o[0] + r.Float64()*400 - 200, o[1] + r.Float64()*400 - 200})
		}
		// Rest positions on the seams of every tile and on texel centres.
		for _, c := range cs {
			cell := c.tile / float64(c.n)
			base := math.Floor(o[0]/c.tile) * c.tile
			rest = append(rest, [2]float64{base + c.tile, o[1] + 3}, [2]float64{base + c.tile - 1e-3, o[1] - 3},
				[2]float64{o[0] + 2, math.Floor(o[1]/c.tile)*c.tile + c.tile}, [2]float64{base + 0.5*cell, o[1]},
				[2]float64{base + c.tile - 0.5*cell, o[1] + 1})
		}
		var om, oz [4]float64
		for c := range cs {
			om[c] = math.Mod(math.Mod(o[0], cs[c].tile)+cs[c].tile, cs[c].tile)
			oz[c] = math.Mod(math.Mod(o[1], cs[c].tile)+cs[c].tile, cs[c].tile)
		}
		var pts []any
		var want []float64
		for _, p := range rest {
			qx, qz, h, det := samplerSurface(cs, p[0], p[1])
			if det < 0.4 { // folded or nearly so: the inverse is not unique
				continue
			}
			pts = append(pts, []float64{qx - o[0], qz - o[1], 0, 0})
			want = append(want, h)
		}
		if len(pts) < 30 {
			t.Fatalf("origin %v: only %d usable points", o, len(pts))
		}
		out := runOceanProbeWith(t, cs, lambda, tm, step, pts, om, oz, run.Options{})
		var worst float64
		for i := range pts {
			if e := math.Abs(out[i].([]float64)[0] - want[i]); e > worst {
				worst = e
			}
		}
		t.Logf("origin %v: %d points, max |kernel - drawn surface| = %.3g m", o, len(pts), worst)
		if worst > worstH {
			worstH = worst
		}
		if worst > 2e-3 {
			t.Errorf("origin %v: kernel differs from the drawn surface by up to %.3g m", o, worst)
		}
	}
	t.Logf("worst height error against the sampler reference: %.3g m", worstH)
}
