package stdlib

import (
	"math"
	"testing"
)

// modalWake advances a real field h0 (at rest) to time t exactly in the
// continuous-time linear problem: every Fourier mode oscillates as
// cos(omega t) with omega = omegaOf(|k|). It uses the same FFT convention and
// the same zeroed modes (k = 0 and the Nyquist lines) as wake_symbol.
func modalWake(n int, length float64, h0 []float64, tm float64, omegaOf func(k float64) float64) []float64 {
	in := make([]any, n*n)
	for i := range in {
		in[i] = []float64{h0[i], 0, 0, 0}
	}
	u := fftInverseGo(n, in) // T[h0] = conj(F)
	g := make([]any, n*n)
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			sx, sz := x, z
			if sx > n/2 {
				sx -= n
			}
			if sz > n/2 {
				sz -= n
			}
			c := 1.0
			if x != n/2 && z != n/2 {
				c = math.Cos(omegaOf(2*math.Pi/length*math.Hypot(float64(sx), float64(sz))) * tm)
			}
			v := u[z*n+x].([]float64)
			// F cos = conj(T[h0]) cos, and the next transform wants F, so conjugate back
			// through the same trick: T[conj(a)] = conj(T[a]) of the reflected field;
			// instead scale conj(u) and apply T, then divide by n^2.
			g[z*n+x] = []float64{c * v[0] / float64(n*n), -c * v[1] / float64(n*n), 0, 0}
		}
	}
	q := fftInverseGo(n, g)
	out := make([]float64, n*n)
	for i := range out {
		out[i] = q[i].([]float64)[0]
	}
	return out
}

func relL2(a, b []float64) float64 {
	var num, den float64
	for i := range a {
		num += (a[i] - b[i]) * (a[i] - b[i])
		den += b[i] * b[i]
	}
	return math.Sqrt(num / den)
}

// A Gaussian bump on still water spreads as deep-water waves: omega^2 = g |k|,
// so long waves outrun short ones. The kernels (Elio FFT, symbol, source, step)
// match the exact modal solution; a non-dispersive wave equation does not.
func TestWakeDispersionGaussianBump(t *testing.T) {
	const n, steps = 32, 100
	r := newWakeRig(t, n, 1)
	r.gravity, r.dt, r.elioFFT = 1, 0.1, true
	h0 := make([]float64, n*n)
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			dx, dz := float64(x-n/2), float64(z-n/2)
			h0[z*n+x] = math.Exp(-(dx*dx + dz*dz) / (2 * 1 * 1))
		}
	}
	// Start at rest to second order: h_prev = h - dt^2 g Q[h] / 2 (so that the first
	// leapfrog step reproduces h(dt) = h - dt^2 g Q[h] / 2).
	qh := applyQ(t, n, float64(n)*r.cell, h0)
	for i := range h0 {
		r.state[i] = [4]float64{h0[i], h0[i] - 0.5*r.dt*r.dt*r.gravity*qh[i], 0, 0}
	}
	for s := 0; s < steps; s++ {
		r.step(hullState{}, 0, 0)
	}
	tm := float64(steps) * r.dt
	got := make([]float64, n*n)
	for i := range got {
		got[i] = r.state[i][0]
	}
	length := float64(n) * r.cell
	exact := modalWake(n, length, h0, tm, func(k float64) float64 { return math.Sqrt(r.gravity * k) })
	errDisp := relL2(got, exact)
	best, bestC := math.Inf(1), 0.0
	for c := 0.2; c <= 2.0; c += 0.02 {
		wave := modalWake(n, length, h0, tm, func(k float64) float64 { return c * k })
		if e := relL2(got, wave); e < best {
			best, bestC = e, c
		}
	}
	t.Logf("after %.0f s: relative L2 error %.4f against the deep-water solution; best non-dispersive wave equation (c = %.2f) %.3f", tm, errDisp, bestC, best)
	if errDisp > 0.02 {
		t.Errorf("error against omega^2 = g k is %g, want <= 0.02", errDisp)
	}
	if best < 5*errDisp || best < 0.2 {
		t.Errorf("a non-dispersive wave equation fits almost as well (%g): the test does not tell dispersion apart", best)
	}

	// Long waves lead. Along a ray the outermost zero crossing is carried by the
	// longest wave the window holds, k = 2 pi / length, at its group distance
	// cg t = (1/2) sqrt(g / k) t; crest spacing grows outward.
	var cross []float64
	row := n / 2
	for x := n / 2; x < n-1; x++ {
		a, b := got[row*n+x], got[row*n+x+1]
		if a*b < 0 {
			cross = append(cross, float64(x-n/2)+a/(a-b))
		}
	}
	t.Logf("zero crossings along +x at r = %.1f cells", cross)
	kMin := 2 * math.Pi / length
	front := 0.5 * math.Sqrt(r.gravity/kMin) * tm
	if len(cross) < 3 {
		t.Fatalf("only %d zero crossings along the ray", len(cross))
	}
	if lead := cross[len(cross)-1]; math.Abs(lead-front) > 0.2*front {
		t.Errorf("leading crossing at r = %.1f, the longest wave's group distance is %.1f", lead, front)
	}
	for i := 0; i+2 < len(cross); i++ {
		if cross[i+2]-cross[i+1] < cross[i+1]-cross[i] {
			t.Errorf("crest spacing shrinks outward at crossing %d: waves are not dispersive in the right order", i)
		}
	}
}

// The leapfrog is stable below dt omega_max = 2 with omega_max^2 = g |k|_max
// (the Nyquist lines are zeroed, so |k|_max = 2 pi sqrt(2) (n/2 - 1) / length)
// and unstable above it.
func TestWakeStabilityLimit(t *testing.T) {
	const n = 16
	cell := 2.0
	length := float64(n) * cell
	kmax := 2 * math.Pi * math.Sqrt2 * (n/2 - 1) / length
	omegaMax := math.Sqrt(9.81 * kmax)
	limit := 2 / omegaMax
	energy := func(factor float64) (start, end float64) {
		r := newWakeRig(t, n, cell)
		r.dt = factor * limit
		for i := range r.state {
			v := math.Sin(float64(i*i%97)) * 0.1
			r.state[i] = [4]float64{v, v, 0, 0}
			start += v * v
		}
		for s := 0; s < 150; s++ {
			r.step(hullState{}, 0, 0)
		}
		for _, c := range r.state {
			end += c[0] * c[0]
		}
		return
	}
	s1, e1 := energy(0.9)
	s2, e2 := energy(1.15)
	t.Logf("omega_max %.3f rad/s, dt limit %.4f s: at 0.9 limit sum h^2 %.4g -> %.4g; at 1.15 limit %.4g -> %.4g", omegaMax, limit, s1, e1, s2, e2)
	if math.IsNaN(e1) || e1 > 20*s1 {
		t.Errorf("unstable below the limit: %g -> %g", s1, e1)
	}
	if !(e2 > 1e6*s2) {
		t.Errorf("no blow-up above the limit: %g -> %g", s2, e2)
	}
}

// With damping and a moving source for many steps the surface stays finite and
// bounded, and after the source stops the energy decays.
func TestWakeEnergyBoundedWithDamping(t *testing.T) {
	const n = 16
	r := newWakeRig(t, n, 4)
	r.dt, r.alpha, r.sponge, r.rate = 0.1, 0.05, 3, 0.5
	r.srcLength, r.srcBeam = 16, 12
	energy := func() (e float64) {
		for _, c := range r.state {
			e += c[0] * c[0]
		}
		return
	}
	var peak float64
	hull := hullState{u: 30, v: 32, heading: math.Pi / 2, speed: 6, depth: 0.5}
	for s := 0; s < 300; s++ {
		hull.u = math.Mod(hull.u+hull.speed*r.dt, float64(n)*r.cell)
		r.step(hull, 0, 0)
		e := energy()
		if math.IsNaN(e) || math.IsInf(e, 0) {
			t.Fatalf("step %d: energy %g", s, e)
		}
		peak = math.Max(peak, e)
	}
	forced := energy()
	hull.speed, hull.depth = 0, 0
	for s := 0; s < 300; s++ {
		r.step(hull, 0, 0)
	}
	free := energy()
	t.Logf("sum h^2: peak %.4g, end of forcing %.4g, 30 s after it stops %.4g", peak, forced, free)
	if peak > 1e3 {
		t.Errorf("energy %g is not bounded by the forcing", peak)
	}
	if free > 0.5*forced {
		t.Errorf("energy does not decay after the source stops: %g -> %g", forced, free)
	}
}
