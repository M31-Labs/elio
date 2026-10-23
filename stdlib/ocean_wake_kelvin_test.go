package stdlib

import (
	"math"
	"sync"
	"testing"
)

// kelvinHalfAngle measures the half-angle (degrees) of the wake's outer energy
// edge from a window field h (h[z*n+x], x increasing ahead of the hull) with
// the hull at window cell (hullX, n/2). For each distance d behind the hull it
// takes the energy of h^2 summed along x over one transverse wavelength, smooths
// it across the track, finds the outermost local maximum above 15% of the
// cross-section's peak, and the cross-track position beyond it where the energy
// falls to 45% of that lobe's peak. That 45% point marks the cusp: the Airy
// function Ai(0)^2 is 45% of its first maximum, so the edge sits at the
// geometric caustic whatever the smearing at finite distance. The half-angle is
// atan of the least-squares slope (with intercept) of that edge against d over
// d in [dMin, dMax]. margin is the sponge width in cells, excluded at the
// window edges. It returns the angle and the number of cross-sections used.
func kelvinHalfAngle(h []float64, n int, hullX, lambdaT, dMin, dMax float64, margin int) (float64, int) {
	half := int(math.Max(3, math.Round(lambdaT/2)))
	var ds, ys []float64
	for d := dMin; d <= dMax; d += 3 {
		xi := int(hullX - d)
		if xi < margin+half {
			break
		}
		e := make([]float64, n)
		for z := n/2 + 1; z < n-margin; z++ {
			for dd := -half; dd <= half; dd++ {
				e[z] += h[z*n+xi+dd] * h[z*n+xi+dd]
			}
		}
		s := make([]float64, n)
		peak := 0.0
		for z := n/2 + 3; z < n-margin-2; z++ {
			s[z] = (e[z-1] + 2*e[z] + e[z+1]) / 4
			peak = math.Max(peak, s[z])
		}
		pz := -1
		for z := n - margin - 3; z > n/2+3; z-- {
			if s[z] >= 0.15*peak && s[z] >= s[z-1] && s[z] >= s[z+1] {
				pz = z
				break
			}
		}
		if pz < 0 {
			continue
		}
		edge := float64(pz)
		for z := pz; z < n-margin-2; z++ {
			if s[z+1] < 0.45*s[pz] {
				edge = float64(z) + (s[z]-0.45*s[pz])/(s[z]-s[z+1])
				break
			}
		}
		ds = append(ds, d)
		ys = append(ys, edge-float64(n/2))
	}
	var sd, sy, sdd, sdy float64
	m := float64(len(ds))
	for i := range ds {
		sd += ds[i]
		sy += ys[i]
		sdd += ds[i] * ds[i]
		sdy += ds[i] * ys[i]
	}
	slope := (m*sdy - sd*sy) / (m*sdd - sd*sd)
	return math.Atan(slope) * 180 / math.Pi, len(ds)
}

// kelvinScenario is the sim-unit setup: a window of n cells, cell = 1 and
// g = 1 (so lengths are in cells and a speed v is in sqrt(g cell) units), a
// 6 x 4 cell pressure patch, dt = 0.5, light damping and a 12-cell sponge.
// With g = 1 the transverse wavelength is 2 pi v^2 cells.
const (
	kelvinN      = 128
	kelvinSponge = 12
	kelvinDT     = 0.5
	kelvinSteps  = 320 // 160 s
)

// referenceWake is an independent plain-Go implementation of the same model
// with the exact operator |k|: the window is a plain array that slides.
func referenceWake(v float64) []float64 {
	n := kelvinN
	h, hp := make([]float64, n*n), make([]float64, n*n)
	hullX, x0 := float64(n-kelvinSponge-10), float64(n-kelvinSponge-10)
	for step := 0; step < kelvinSteps; step++ {
		depth := 0.5 * math.Min(1, float64(step)*kelvinDT/30)
		s := make([]float64, n*n)
		in := make([]any, n*n)
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				u, w := (float64(x)+0.5-hullX)/3, (float64(z)+0.5-float64(n)/2)/2
				q := u*u + w*w
				head := 0.0
				if q < 1 {
					head = depth * (1 - q) * (1 - q)
				}
				s[z*n+x] = h[z*n+x] + head
				in[z*n+x] = []float64{s[z*n+x], 0, 0, 0}
			}
		}
		q := fftInverseGo(n, symbolGo(n, float64(n), fftInverseGo(n, in)))
		next := make([]float64, n*n)
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				e := min(min(x, z), min(n-1-x, n-1-z))
				tt := sat((kelvinSponge - float64(e)) / kelvinSponge)
				beta := 1 - (0.01+0.8*tt*tt)*kelvinDT
				next[z*n+x] = h[z*n+x] + beta*(h[z*n+x]-hp[z*n+x]) - kelvinDT*kelvinDT*q[z*n+x].([]float64)[0]
			}
		}
		hp, h = h, next
		hullX += v * kelvinDT
		for hullX > x0+1 {
			for _, a := range [][]float64{h, hp} {
				for z := 0; z < n; z++ {
					copy(a[z*n:z*n+n-1], a[z*n+1:z*n+n])
					a[z*n+n-1] = 0
				}
			}
			hullX--
		}
	}
	return h
}

// symbolGo is wake_symbol in Go: conj(u) |k| / n^2 with the zeroed lines.
func symbolGo(n int, length float64, u []any) []any {
	out := make([]any, n*n)
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			sx, sz := x, z
			if sx > n/2 {
				sx -= n
			}
			if sz > n/2 {
				sz -= n
			}
			gain := 0.0
			if x != n/2 && z != n/2 {
				gain = 2 * math.Pi / length * math.Hypot(float64(sx), float64(sz)) / float64(n*n)
			}
			c := u[z*n+x].([]float64)
			out[z*n+x] = []float64{gain * c[0], -gain * c[1], 0, 0}
		}
	}
	return out
}

// The metric itself: with the exact operator the energy edge gives 19.47
// degrees (1 / (2 sqrt 2) = tan 19.47) to within 3 degrees at two speeds.
func TestKelvinMetricOnExactOperator(t *testing.T) {
	var wg sync.WaitGroup
	for _, v := range []float64{1.2, 1.5} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h := referenceWake(v)
			ang, n := kelvinHalfAngle(h, kelvinN, float64(kelvinN-kelvinSponge-10), 2*math.Pi*v*v, 14, 60, kelvinSponge)
			t.Logf("exact operator, v = %.1f sqrt(g cell): half-angle %.2f deg from %d cross-sections (transverse wavelength %.1f cells)", v, ang, n, 2*math.Pi*v*v)
			if math.Abs(ang-19.47) > 3 || n < 10 {
				t.Errorf("v = %.1f: half-angle %.2f deg (n = %d), want 19.47 +- 3", v, ang, n)
			}
		}()
	}
	wg.Wait()
}

// The same scenario through the Elio kernels on the interpreter: the window
// follows the hull by whole-cell shifts of the toroidal buffer.
func TestKelvinWedgeOnKernels(t *testing.T) {
	if testing.Short() {
		t.Skip("runs 320 steps of the kernels on a 128 x 128 window, twice")
	}
	for _, v := range []float64{1.2, 1.5} {
		t.Run("", func(t *testing.T) {
			t.Parallel()
			n := kelvinN
			r := newWakeRig(t, n, 1)
			r.gravity, r.dt, r.alpha, r.sponge, r.rate, r.tau = 1, kelvinDT, 0.01, kelvinSponge, 0.8, 1e9
			r.srcLength, r.srcBeam = 6, 4
			x0 := float64(n - kelvinSponge - 10)
			hull := hullState{u: x0, v: float64(n) / 2, heading: math.Pi / 2, speed: v}
			shifts := 0
			for step := 0; step < kelvinSteps; step++ {
				hull.depth = 0.5 * math.Min(1, float64(step)*kelvinDT/30)
				hull.u += v * kelvinDT
				shift := 0
				for hull.u > x0+1 {
					hull.u--
					shift++
				}
				shifts += shift
				r.step(hull, shift, 0)
			}
			ang, m := kelvinHalfAngle(r.window(), n, hull.u, 2*math.Pi*v*v, 14, 60, kelvinSponge)
			t.Logf("kernels, v = %.1f sqrt(g cell): half-angle %.2f deg from %d cross-sections after %d window shifts", v, ang, m, shifts)
			if math.Abs(ang-19.47) > 3 || m < 10 {
				t.Errorf("v = %.1f: half-angle %.2f deg (n = %d), want 19.47 +- 3", v, ang, m)
			}
		})
	}
}
