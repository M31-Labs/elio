package stdlib

import (
	"math"
	"math/bits"
	"math/cmplx"
	"math/rand"
	"testing"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/run"
)

// hermitianPair returns n*n elements whose two slots are each Hermitian:
// H[(-kz) mod n, (-kx) mod n] = conj(H[kz, kx]). The self-conjugate bins (kz
// and kx each 0 or n/2) are real. Each slot is the spectrum of a real field.
func hermitianPair(n int, seed int64) [][2]complex128 {
	r := rand.New(rand.NewSource(seed))
	v := make([][2]complex128, n*n)
	neg := func(k int) int { return (n - k) % n }
	for kz := 0; kz < n; kz++ {
		for kx := 0; kx < n; kx++ {
			i, j := kz*n+kx, neg(kz)*n+neg(kx)
			if j < i {
				continue // already set as the conjugate of its partner
			}
			for s := 0; s < 2; s++ {
				c := complex(r.Float64()*2-1, r.Float64()*2-1)
				if i == j {
					c = complex(real(c), 0)
				}
				v[i][s] = c
				v[j][s] = cmplx.Conj(c)
			}
		}
	}
	return v
}

func fftOnInterpreter(t *testing.T, form string, n int, in [][2]complex128) [][2]complex128 {
	t.Helper()
	data := packVec4(in)
	switch form {
	case "compute":
		m, err := FFT2DCompute(n)
		if err != nil {
			t.Fatal(err)
		}
		mem := &run.Memory{Vars: map[string]any{"data": data}}
		for _, k := range []string{"fft_rows", "fft_cols"} {
			if err := run.Run(m, k, n*(n/2), mem); err != nil {
				t.Fatal(err)
			}
		}
		return unpackVec4(data)
	default:
		m, err := FFT2DGather(n)
		if err != nil {
			t.Fatal(err)
		}
		return runGather(t, m, n, data)
	}
}

func runGather(t *testing.T, m *ir.Module, n int, src []any) [][2]complex128 {
	t.Helper()
	dst := make([]any, n*n)
	for i := range dst {
		dst[i] = []float64{0, 0, 0, 0}
	}
	for axis := int64(0); axis < 2; axis++ {
		for stage := 0; stage < bits.TrailingZeros(uint(n)); stage++ {
			mem := &run.Memory{Vars: map[string]any{
				"fft": map[string]any{"n": int64(n), "stage": int64(stage), "axis": axis, "_pad": int64(0)},
				"src": src, "dst": dst,
			}}
			if err := run.Run(m, "fft_stage", n*n, mem); err != nil {
				t.Fatal(err)
			}
			src, dst = dst, src
		}
	}
	return unpackVec4(src)
}

// TestFFTTwoRealSignalsPerPass checks the packing the ocean relies on: two
// Hermitian spectra in the two slots invert to two real fields, so the
// imaginary parts of the output vanish while the real parts match the DFT.
func TestFFTTwoRealSignalsPerPass(t *testing.T) {
	for _, form := range []string{"compute", "gather"} {
		for _, n := range []int{8, 32} {
			in := hermitianPair(n, int64(7+n))
			out := fftOnInterpreter(t, form, n, in)
			var re2, imMax float64
			for _, e := range out {
				for s := 0; s < 2; s++ {
					re2 += real(e[s]) * real(e[s])
					imMax = math.Max(imMax, math.Abs(imag(e[s])))
				}
			}
			realRMS := math.Sqrt(re2 / float64(2*len(out)))
			t.Logf("%s n=%d: real RMS %.3g, max |imag| %.3g", form, n, realRMS, imMax)
			if realRMS < 1e-3 {
				t.Fatalf("%s n=%d: output is trivially zero (real RMS %g)", form, n, realRMS)
			}
			if imMax > 1e-9*realRMS {
				t.Errorf("%s n=%d: max |imag| %g exceeds 1e-9 of real RMS %g", form, n, imMax, realRMS)
			}
			if e := maxErr(out, dft2DInverse(in, n)); e > 1e-9*realRMS*float64(n) {
				t.Errorf("%s n=%d: differs from DFT by %g", form, n, e)
			}
		}
	}
}
