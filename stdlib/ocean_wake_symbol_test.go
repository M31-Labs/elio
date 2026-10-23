package stdlib

import (
	"math"
	"math/bits"
	"testing"

	"m31labs.dev/elio/run"
)

// fftInverseGather runs the Elio gather inverse FFT (unnormalized, e^{+i})
// over a packed vec4 buffer, as the graph does: rows then columns, one pass per
// stage, swapping buffers.
func fftInverseGather(t *testing.T, n int, buf []any) []any {
	t.Helper()
	m, err := FFT2DGather(n)
	if err != nil {
		t.Fatal(err)
	}
	src, dst := buf, make([]any, n*n)
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
	return src
}

// runWakeSymbol runs wake_symbol on a window of n x n cells, length metres wide.
func runWakeSymbol(t *testing.T, n int, length float64, in []any) []any {
	t.Helper()
	m := checkedWakeKernel(t, "wake_symbol")
	out := make([]any, n*n)
	for i := range out {
		out[i] = []float64{9, 9, 9, 9}
	}
	vars := map[string]any{
		"sym":   map[string]any{"n": int64(n), "_pad0": int64(0), "_pad1": int64(0), "_pad2": int64(0), "length": length, "_pad3": 0.0, "_pad4": 0.0, "_pad5": 0.0},
		"fftIn": in, "fftOut": out,
	}
	if err := run.Run(m, "wake_symbol", n*n, &run.Memory{Vars: vars}); err != nil {
		t.Fatal(err)
	}
	return out
}

// applyQ applies Q = |k| to a real field with the Elio kernels: inverse FFT,
// wake_symbol, inverse FFT, real part.
func applyQ(t *testing.T, n int, length float64, s []float64) []float64 {
	t.Helper()
	in := make([]any, n*n)
	for i := range in {
		in[i] = []float64{s[i], 0, 0, 0}
	}
	u := fftInverseGather(t, n, in)
	g := runWakeSymbol(t, n, length, u)
	q := fftInverseGather(t, n, g)
	out := make([]float64, n*n)
	for i := range out {
		out[i] = q[i].([]float64)[0]
	}
	return out
}

func TestWakeSymbolSineMode(t *testing.T) {
	const n, length = 16, 64.0
	for _, mode := range [][2]int{{1, 0}, {0, 3}, {2, 2}, {-3, 1}} {
		k := 2 * math.Pi / length * math.Hypot(float64(mode[0]), float64(mode[1]))
		s := make([]float64, n*n)
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				s[z*n+x] = math.Cos(2 * math.Pi * (float64(mode[0]*x) + float64(mode[1]*z)) / n)
			}
		}
		q := applyQ(t, n, length, s)
		var worst float64
		for i := range q {
			worst = math.Max(worst, math.Abs(q[i]-k*s[i]))
		}
		t.Logf("mode %v: |k| = %.4f rad/m, max error %.2g", mode, k, worst)
		if worst > 1e-12 {
			t.Errorf("mode %v: Q[cos] differs from |k| cos by %g", mode, worst)
		}
	}
}

// Constant fields, and the Nyquist lines, are in the null space: Q zeroes k = 0
// and the Nyquist row and column, as the ocean does.
func TestWakeSymbolNullSpace(t *testing.T) {
	const n, length = 16, 64.0
	cst := make([]float64, n*n)
	nyq := make([]float64, n*n)
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			cst[z*n+x] = 3
			nyq[z*n+x] = float64(1 - 2*(x%2))
		}
	}
	for name, f := range map[string][]float64{"constant": cst, "x Nyquist": nyq} {
		for i, v := range applyQ(t, n, length, f) {
			if math.Abs(v) > 1e-12 {
				t.Fatalf("%s: Q[%d] = %g, want 0", name, i, v)
			}
		}
	}
}

// A localized bump against a direct DFT reference: Q[s] = IDFT(|k| DFT(s)).
func TestWakeSymbolMatchesDFT(t *testing.T) {
	const n, length = 16, 48.0
	s := make([]float64, n*n)
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			dx, dz := float64(posMod(x+8, n)-8), float64(posMod(z+8, n)-8)
			s[z*n+x] = math.Exp(-(dx*dx+(dz-1)*(dz-1))/6) + 0.1*float64((x*7+z*3)%5)
		}
	}
	got := applyQ(t, n, length, s)
	var worst, scale float64
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			var sum float64
			for kz := 0; kz < n; kz++ {
				for kx := 0; kx < n; kx++ {
					sx, sz := kx, kz
					if sx > n/2 {
						sx -= n
					}
					if sz > n/2 {
						sz -= n
					}
					if kx == n/2 || kz == n/2 {
						continue
					}
					k := 2 * math.Pi / length * math.Hypot(float64(sx), float64(sz))
					var fr, fi float64 // F(k) = sum s e^{-i k.x}
					for zz := 0; zz < n; zz++ {
						for xx := 0; xx < n; xx++ {
							a := -2 * math.Pi * float64(kx*xx+kz*zz) / n
							fr, fi = fr+s[zz*n+xx]*math.Cos(a), fi+s[zz*n+xx]*math.Sin(a)
						}
					}
					a := 2 * math.Pi * float64(kx*x+kz*z) / n
					sum += k * (fr*math.Cos(a) - fi*math.Sin(a))
				}
			}
			sum /= n * n
			worst = math.Max(worst, math.Abs(got[z*n+x]-sum))
			scale = math.Max(scale, math.Abs(sum))
		}
	}
	t.Logf("max error %.2g against a field of scale %.3g", worst, scale)
	if worst > 1e-10*scale {
		t.Errorf("Q differs from the DFT reference by %g (scale %g)", worst, scale)
	}
}
