package stdlib

import (
	"math"
	"testing"
)

// TestOceanFieldTravelsTowardWindDir checks the direction convention end to end:
// windDir and swellDir are the angle theta of the direction the waves travel
// toward, vector (cos theta, sin theta) in world (x, z). With a narrow spread,
// the height field at t + dt is the field at t shifted along that vector. The
// shift is found as the integer circular shift that maximizes the
// cross-correlation of the two frames (cascade 1, n = 64, 3.9 m texels).
func TestOceanFieldTravelsTowardWindDir(t *testing.T) {
	for _, theta := range []float64{0, math.Pi / 2, math.Pi, -math.Pi / 4} {
		s := defaultSpec(1, 64)
		s.spread, s.windDir, s.swellDir, s.seed = 32, theta, theta, 5
		h0, omega := runOceanSpectrum(t, s)
		a := runOceanEvolve(t, s, h0, omega, 100)[0]
		b := runOceanEvolve(t, s, h0, omega, 101)[0]
		n := s.n
		best, bx, bz := math.Inf(-1), 0, 0
		for sz := -8; sz <= 8; sz++ {
			for sx := -8; sx <= 8; sx++ {
				// a shifted by (sx, sz) against b: b(x) ~ a(x - s).
				var c float64
				for z := 0; z < n; z++ {
					for x := 0; x < n; x++ {
						c += a[((z-sz+n)%n)*n+(x-sx+n)%n] * b[z*n+x]
					}
				}
				if c > best {
					best, bx, bz = c, sx, sz
				}
			}
		}
		dx, dz := math.Cos(theta), math.Sin(theta)
		along := float64(bx)*dx + float64(bz)*dz
		across := float64(bx)*dz - float64(bz)*dx
		t.Logf("theta %.3f rad: crests moved (%d, %d) texels in 1 s, %.1f along and %.1f across the wind vector", theta, bx, bz, along, across)
		if along < 1 || math.Abs(across) > 1 {
			t.Errorf("theta %.3f: shift (%d, %d) is not along (%.2f, %.2f)", theta, bx, bz, dx, dz)
		}
	}
}
