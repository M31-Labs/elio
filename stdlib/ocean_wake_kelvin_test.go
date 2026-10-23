package stdlib

import (
	"math"
	"sync"
	"testing"

	"m31labs.dev/elio/internal/wakesim"
)

// The metric itself: with the exact operator the energy edge gives 19.47
// degrees (1 / (2 sqrt 2) = tan 19.47) to within 3 degrees at two speeds.
func TestKelvinMetricOnExactOperator(t *testing.T) {
	var wg sync.WaitGroup
	for _, v := range []float64{1.2, 1.5} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h := wakesim.Reference(v, 6, 4)
			ang, n := wakesim.HalfAngle(h, wakesim.N, float64(wakesim.N-wakesim.Sponge-10), 2*math.Pi*v*v, 14, 60, wakesim.Sponge)
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
			n := wakesim.N
			r := newWakeRig(t, n, 1)
			r.gravity, r.dt, r.alpha, r.sponge, r.rate, r.tau = 1, wakesim.DT, 0.01, wakesim.Sponge, 0.8, 1e9
			r.srcLength, r.srcBeam = 6, 4
			x0 := float64(n - wakesim.Sponge - 10)
			hull := hullState{u: x0, v: float64(n) / 2, heading: math.Pi / 2, speed: v}
			shifts := 0
			for step := 0; step < wakesim.Steps; step++ {
				hull.depth = 0.5 * math.Min(1, float64(step)*wakesim.DT/30)
				hull.u += v * wakesim.DT
				shift := 0
				for hull.u > x0+1 {
					hull.u--
					shift++
				}
				shifts += shift
				r.step(hull, shift, 0)
			}
			ang, m := wakesim.HalfAngle(r.window(), n, hull.u, 2*math.Pi*v*v, 14, 60, wakesim.Sponge)
			t.Logf("kernels, v = %.1f sqrt(g cell): half-angle %.2f deg from %d cross-sections after %d window shifts", v, ang, m, shifts)
			if math.Abs(ang-19.47) > 3 || m < 10 {
				t.Errorf("v = %.1f: half-angle %.2f deg (n = %d), want 19.47 +- 3", v, ang, m)
			}
		})
	}
}
