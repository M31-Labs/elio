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
	u, v := px/c.tile, pz/c.tile
	fx, fz := (u-math.Floor(u))*float64(c.n), (v-math.Floor(v))*float64(c.n)
	x0, z0 := int(math.Floor(fx)), int(math.Floor(fz))
	if x0 > c.n-1 {
		x0 = c.n - 1
	}
	if z0 > c.n-1 {
		z0 = c.n - 1
	}
	rx, rz := fx-math.Floor(fx), fz-math.Floor(fz)
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
