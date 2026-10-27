package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// pcgRef is the reference PCG output hash (Jarzynski and Olano 2020), in uint32.
func pcgRef(v uint32) uint32 {
	state := v*747796405 + 2891336453
	word := ((state >> ((state >> 28) + 4)) ^ state) * 277803737
	return (word >> 22) ^ word
}

// oceanNormalsRef is the Go reference of the ocean RNG block: the normals and
// uniforms for the key (seed, cascade, ix, iz), with signed ix and iz.
func oceanNormalsRef(seed, cascade uint32, ix, iz int) (z0, z1, u1, u2 float64) {
	h := uint32(0)
	words := []uint32{seed, cascade, uint32(int32(ix)), uint32(int32(iz)), 2654435769, 2246822519}
	var b1, b2 uint32
	for r, w := range words {
		h = pcgRef(h ^ w)
		if r == 4 {
			b1 = h
		}
		if r == 5 {
			b2 = h
		}
	}
	u1 = (float64(b1>>8) + 1) / 16777216
	u2 = (float64(b2>>8) + 1) / 16777216
	r := math.Sqrt(-2 * math.Log(u1))
	return r * math.Cos(2*math.Pi*u2), r * math.Sin(2*math.Pi*u2), u1, u2
}

// runOceanRng runs ocean_rng over an n x n grid and returns the vec4 outputs
// (z0, z1, u1, u2) in texel order.
func runOceanRng(t *testing.T, n int, seed, cascade int64) [][]float64 {
	t.Helper()
	m, err := OceanRng()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	out := make([]any, n*n)
	for i := range out {
		out[i] = []float64{0, 0, 0, 0}
	}
	mem := &run.Memory{Vars: map[string]any{
		"rng":     map[string]any{"n": int64(n), "cascade": cascade, "seed": seed, "_pad": int64(0)},
		"normals": out,
	}}
	if err := run.Run(m, "ocean_rng", n*n, mem); err != nil {
		t.Fatal(err)
	}
	res := make([][]float64, n*n)
	for i := range out {
		res[i] = out[i].([]float64)
	}
	return res
}

func signedBin(x, n int) int {
	if x < n/2 {
		return x
	}
	return x - n
}

func TestOceanRngMatchesReference(t *testing.T) {
	n := 64
	for _, seed := range []int64{0, 7, 4294967295} {
		got := runOceanRng(t, n, seed, 2)
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				z0, z1, u1, u2 := oceanNormalsRef(uint32(seed), 2, signedBin(x, n), signedBin(z, n))
				g := got[z*n+x]
				for k, w := range []float64{z0, z1, u1, u2} {
					if math.Abs(g[k]-w) > 1e-12 {
						t.Fatalf("seed %d bin (%d,%d) lane %d: got %.17g want %.17g", seed, x, z, k, g[k], w)
					}
				}
			}
		}
	}
}

func TestOceanRngStatistics(t *testing.T) {
	n := 256
	got := runOceanRng(t, n, 12345, 1)
	var sum, sum2, sumU, sumLag, sumCross float64
	var count float64
	for i, g := range got {
		for _, v := range g[:2] {
			sum += v
			sum2 += v * v
			count++
		}
		sumU += g[2] + g[3]
		sumCross += g[0] * g[1]
		if i+1 < len(got) {
			sumLag += g[0] * got[i+1][0]
		}
		if g[2] <= 0 || g[2] > 1 || g[3] <= 0 || g[3] > 1 {
			t.Fatalf("uniform outside (0,1]: %v", g[2:])
		}
	}
	mean := sum / count
	variance := sum2/count - mean*mean
	corr := sumCross / float64(len(got))
	lag := sumLag / float64(len(got)-1)
	t.Logf("RNG n=%d draws=%d normals=%d: mean %.5f variance %.5f uniform mean %.5f corr(z0,z1) %.5f lag-1 corr %.5f",
		n, len(got), int(count), mean, variance, sumU/(2*float64(len(got))), corr, lag)
	if math.Abs(mean) > 0.01 {
		t.Errorf("mean %g outside 0.01 of 0", mean)
	}
	if math.Abs(variance-1) > 0.02 {
		t.Errorf("variance %g outside 0.02 of 1", variance)
	}
	if math.Abs(corr) > 0.02 || math.Abs(lag) > 0.02 {
		t.Errorf("correlation too large: pair %g lag %g", corr, lag)
	}
	if math.Abs(sumU/(2*float64(len(got)))-0.5) > 0.01 {
		t.Errorf("uniform mean %g not near 0.5", sumU/(2*float64(len(got))))
	}
}

// TestOceanRngIsResolutionIndependent checks the key is the signed wavenumber
// index: changing n never reshuffles a wave.
func TestOceanRngIsResolutionIndependent(t *testing.T) {
	small := runOceanRng(t, 64, 99, 3)
	large := runOceanRng(t, 256, 99, 3)
	for iz := -31; iz < 32; iz++ {
		for ix := -31; ix < 32; ix++ {
			xs, zs := (ix+64)%64, (iz+64)%64
			xl, zl := (ix+256)%256, (iz+256)%256
			a, b := small[zs*64+xs], large[zl*256+xl]
			for k := range a {
				if a[k] != b[k] {
					t.Fatalf("bin (%d,%d) differs between n=64 and n=256: %v vs %v", ix, iz, a, b)
				}
			}
		}
	}
}
