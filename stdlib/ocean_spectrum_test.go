package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// oceanSpec mirrors the OceanSpectrumParams uniform block.
type oceanSpec struct {
	n, cascade, seed     int
	windSpeed, fetchKm   float64
	gamma, spread, swell float64
	windDir, swellDir    float64
	tile, kMin, kMax     float64
}

func defaultSpec(cascade, n int) oceanSpec {
	// The four default cascades: tile sizes and band edges from the wave model.
	tiles := []float64{1600, 250, 37, 5.3}
	edges := []float64{2 * math.Pi / 1600, 0.12, 0.80, 5.4, 37}
	return oceanSpec{
		n: n, cascade: cascade, seed: 1,
		windSpeed: 9, fetchKm: 300, gamma: 3.3, spread: 8,
		windDir: 0.3,
		tile:    tiles[cascade], kMin: edges[cascade], kMax: edges[cascade+1],
	}
}

func (s oceanSpec) uniform() map[string]any {
	return map[string]any{
		"n": int64(s.n), "cascade": int64(s.cascade), "seed": int64(s.seed), "_pad": int64(0),
		"windSpeed": s.windSpeed, "fetchKm": s.fetchKm, "gamma": s.gamma, "spread": s.spread,
		"swell": s.swell, "windDir": s.windDir, "swellDir": s.swellDir, "tile": s.tile,
		"kMin": s.kMin, "kMax": s.kMax, "_pad1": 0.0, "_pad2": 0.0,
	}
}

// runOceanSpectrum runs ocean_spectrum and returns h0 (vec4 per bin) and omega.
func runOceanSpectrum(t *testing.T, s oceanSpec) (h0 [][]float64, omega []float64) {
	t.Helper()
	m, err := OceanSpectrum()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	buf := make([]any, s.n*s.n)
	for i := range buf {
		buf[i] = []float64{9, 9, 9, 9} // sentinel: every bin must be written
	}
	om := make([]float64, s.n*s.n)
	mem := &run.Memory{Vars: map[string]any{"spec": s.uniform(), "h0": buf, "omega": om}}
	if err := run.Run(m, "ocean_spectrum", s.n*s.n, mem); err != nil {
		t.Fatal(err)
	}
	h0 = make([][]float64, len(buf))
	for i := range buf {
		h0[i] = buf[i].([]float64)
	}
	return h0, om
}

const oceanG = 9.81

// jonswapRef is the JONSWAP frequency spectrum S(omega) in m^2 s (Hasselmann et
// al. 1973) for the given peak frequency, alpha and gamma.
func jonswapRef(w, wp, alpha, gamma float64) float64 {
	sig := 0.07
	if w > wp {
		sig = 0.09
	}
	r := math.Exp(-(w - wp) * (w - wp) / (2 * sig * sig * wp * wp))
	return alpha * oceanG * oceanG * math.Pow(w, -5) * math.Exp(-1.25*math.Pow(wp/w, 4)) * math.Pow(gamma, r)
}

// oceanDensityRef is the 2D wavenumber spectrum S(kx, kz) in m^4, from the same
// model as the kernel: JONSWAP over omega, Hasselmann-style spreading
// cos^2s(theta/2) with s depending on omega/omega_p, Jacobian g/(2 omega k).
func oceanDensityRef(s oceanSpec, kx, kz float64) float64 {
	k := math.Hypot(kx, kz)
	w := math.Sqrt(oceanG * k)
	fetch := s.fetchKm * 1000
	wp := 22 * math.Pow(oceanG*oceanG/(s.windSpeed*fetch), 1.0/3)
	alpha := 0.076 * math.Pow(s.windSpeed*s.windSpeed/(fetch*oceanG), 0.22)
	var dens float64
	for c := 0; c < 2; c++ {
		weight, wpk, alph, gam, sp, dir := 1-s.swell, wp, alpha, s.gamma, s.spread, s.windDir
		if c == 1 {
			weight, wpk, alph, gam, sp, dir = s.swell, 0.5*wp, alpha*0.0625, 7.0, math.Min(4*s.spread, 64), s.swellDir
		}
		if weight <= 0 {
			continue
		}
		var sw float64
		if w <= wpk {
			sw = sp * math.Pow(w/wpk, 4.06)
		} else {
			sw = sp * math.Pow(w/wpk, -2.33)
		}
		cosT := (kx*math.Cos(dir) + kz*math.Sin(dir)) / k
		base := math.Max(0.5*(1+cosT), 1e-8)
		lg1, _ := math.Lgamma(sw + 1)
		lg2, _ := math.Lgamma(sw + 0.5)
		norm := math.Exp(lg1 - lg2 - math.Log(2*math.Sqrt(math.Pi)))
		dens += weight * jonswapRef(w, wpk, alph, gam) * norm * math.Pow(base, sw) * oceanG / (2 * w * k)
	}
	return dens
}

func TestOceanSpectrumMatchesReference(t *testing.T) {
	cases := []oceanSpec{defaultSpec(1, 64), defaultSpec(3, 64), defaultSpec(0, 32)}
	sw := defaultSpec(2, 64)
	sw.swell, sw.swellDir, sw.seed = 0.4, -1.1, 77
	cases = append(cases, sw)
	for _, s := range cases {
		h0, omega := runOceanSpectrum(t, s)
		n := s.n
		dk := 2 * math.Pi / s.tile
		var nonzero int
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				ix, iz := signedBin(x, n), signedBin(z, n)
				kx, kz := dk*float64(ix), dk*float64(iz)
				k := math.Hypot(kx, kz)
				if want := math.Sqrt(oceanG * k); math.Abs(omega[z*n+x]-want) > 1e-12 {
					t.Fatalf("omega (%d,%d) = %g want %g", ix, iz, omega[z*n+x], want)
				}
				var want [4]float64
				if x != n/2 && z != n/2 && k > 0 && k >= s.kMin && k < s.kMax {
					// h0(k) and conj h0(-k), each (z0 + i z1) * sqrt(S) * dk / 2.
					a0, a1, _, _ := oceanNormalsRef(uint32(s.seed), uint32(s.cascade), ix, iz)
					b0, b1, _, _ := oceanNormalsRef(uint32(s.seed), uint32(s.cascade), -ix, -iz)
					amp := 0.5 * math.Sqrt(oceanDensityRef(s, kx, kz)) * dk
					ampM := 0.5 * math.Sqrt(oceanDensityRef(s, -kx, -kz)) * dk
					want = [4]float64{a0 * amp, a1 * amp, b0 * ampM, -b1 * ampM}
				}
				g := h0[z*n+x]
				for c := 0; c < 4; c++ {
					if d := math.Abs(g[c] - want[c]); d > 1e-7*math.Abs(want[c])+1e-15 {
						t.Fatalf("cascade %d bin (%d,%d) lane %d: got %.12g want %.12g", s.cascade, ix, iz, c, g[c], want[c])
					}
					if g[c] != 0 {
						nonzero++
					}
				}
			}
		}
		if nonzero == 0 {
			t.Errorf("cascade %d: no non-zero bins", s.cascade)
		}
	}
}

func TestOceanSpectrumHermitianAndEmptyLines(t *testing.T) {
	for _, s := range []oceanSpec{defaultSpec(0, 64), defaultSpec(1, 64), defaultSpec(2, 64), defaultSpec(3, 64)} {
		h0, _ := runOceanSpectrum(t, s)
		n := s.n
		dk := 2 * math.Pi / s.tile
		inBand, zeroInBand := 0, 0
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				g := h0[z*n+x]
				m := h0[((n-z)%n)*n+(n-x)%n]
				// The mirror bin holds conj of this bin's h0(k) in its (z, w) lanes.
				if m[2] != g[0] || m[3] != -g[1] {
					t.Fatalf("cascade %d (%d,%d): mirror lanes (%g,%g) are not conj of (%g,%g)", s.cascade, x, z, m[2], m[3], g[0], g[1])
				}
				ix, iz := signedBin(x, n), signedBin(z, n)
				k := dk * math.Hypot(float64(ix), float64(iz))
				allZero := g[0] == 0 && g[1] == 0 && g[2] == 0 && g[3] == 0
				if x == n/2 || z == n/2 || (ix == 0 && iz == 0) {
					if !allZero {
						t.Errorf("cascade %d (%d,%d): Nyquist or k=0 bin is not exactly zero: %v", s.cascade, ix, iz, g)
					}
					continue
				}
				if k < s.kMin || k >= s.kMax {
					if !allZero {
						t.Errorf("cascade %d (%d,%d) |k|=%g outside [%g,%g) is not zero: %v", s.cascade, ix, iz, k, s.kMin, s.kMax, g)
					}
					continue
				}
				inBand++
				if allZero {
					zeroInBand++
				}
			}
		}
		t.Logf("cascade %d n=%d: %d in-band bins, %d zero", s.cascade, n, inBand, zeroInBand)
		if inBand == 0 || zeroInBand > inBand/100 {
			t.Errorf("cascade %d: %d of %d in-band bins are zero", s.cascade, zeroInBand, inBand)
		}
	}
}

// jonswapHsEstimate is the closed-form significant wave height 4 sqrt(m0) with
// m0 = alpha g^2 / omega_p^4 * (0.230 + 0.0336 gamma - 0.185/(1.9 + gamma)),
// Goda's fit to the integral of the JONSWAP spectrum.
func jonswapHsEstimate(u, fetchKm, gamma float64) (hs, m0 float64) {
	fetch := fetchKm * 1000
	wp := 22 * math.Pow(oceanG*oceanG/(u*fetch), 1.0/3)
	alpha := 0.076 * math.Pow(u*u/(fetch*oceanG), 0.22)
	m0 = alpha * oceanG * oceanG / math.Pow(wp, 4) * (0.230 + 0.0336*gamma - 0.185/(1.9+gamma))
	return 4 * math.Sqrt(m0), m0
}

// realizedM0 sums |h(k, t=0)|^2 over every bin of every default cascade, where
// h(k, 0) = h0(k) + conj h0(-k): the variance of the realized height field.
func realizedM0(t *testing.T, base oceanSpec) (m0, expected float64) {
	for c := 0; c < 4; c++ {
		s := defaultSpec(c, 64)
		s.seed, s.windSpeed, s.fetchKm, s.gamma, s.spread, s.windDir = base.seed, base.windSpeed, base.fetchKm, base.gamma, base.spread, base.windDir
		h0, _ := runOceanSpectrum(t, s)
		dk := 2 * math.Pi / s.tile
		for z := 0; z < s.n; z++ {
			for x := 0; x < s.n; x++ {
				g := h0[z*s.n+x]
				re, im := g[0]+g[2], g[1]+g[3]
				m0 += re*re + im*im
				if g[0] != 0 || g[1] != 0 {
					ix, iz := signedBin(x, s.n), signedBin(z, s.n)
					// Expected |h|^2 over seeds is (S(k) + S(-k)) dk^2 / 2.
					kx, kz := dk*float64(ix), dk*float64(iz)
					expected += 0.5 * (oceanDensityRef(s, kx, kz) + oceanDensityRef(s, -kx, -kz)) * dk * dk
				}
			}
		}
	}
	return m0, expected
}

func TestOceanSpectrumSignificantWaveHeight(t *testing.T) {
	base := defaultSpec(0, 64)
	est, estM0 := jonswapHsEstimate(base.windSpeed, base.fetchKm, base.gamma)
	var hsSum float64
	const seeds = 6
	for seed := 1; seed <= seeds; seed++ {
		base.seed = seed
		m0, expected := realizedM0(t, base)
		hs := 4 * math.Sqrt(m0)
		hsSum += hs
		t.Logf("seed %d: Hs %.3f m (JONSWAP estimate %.3f m; bin-sum expectation %.3f m; fetch-law E=1.6e-7 chi gives 2.518 m)",
			seed, hs, est, 4*math.Sqrt(expected))
		if seed == 1 && math.Abs(4*math.Sqrt(expected)-est) > 0.05*est {
			t.Errorf("expected Hs from the bin sum %.3f is not within 5%% of the estimate %.3f", 4*math.Sqrt(expected), est)
		}
		if math.Abs(hs-est) > 0.10*est {
			t.Errorf("seed %d: Hs %.3f m not within 10%% of the JONSWAP estimate %.3f m (m0 %.4f vs %.4f)", seed, hs, est, m0, estM0)
		}
	}
	t.Logf("mean Hs over %d seeds: %.3f m", seeds, hsSum/seeds)
	if mean := hsSum / seeds; math.Abs(mean-est) > 0.05*est {
		t.Errorf("mean Hs %.3f m not within 5%% of the estimate %.3f m", mean, est)
	}
}
