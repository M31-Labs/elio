package stdlib

import (
	"math"
	"math/bits"
	"testing"

	"m31labs.dev/elio/run"
)

// TestFFT2DGatherFloat32 runs the n = 256 gather FFT with every float result
// rounded to float32, as a WebGL2 pass computes it, and compares with the
// float64 DFT of the same (float32-rounded) input. This takes tens of seconds.
func TestFFT2DGatherFloat32(t *testing.T) {
	if testing.Short() {
		t.Skip("n=256 interpreter run takes tens of seconds")
	}
	const n = 256
	m, err := FFT2DGather(n)
	if err != nil {
		t.Fatal(err)
	}
	in := randomSpectrum(n, 7)
	for i := range in {
		for s := 0; s < 2; s++ {
			in[i][s] = complex(float64(float32(real(in[i][s]))), float64(float32(imag(in[i][s]))))
		}
	}
	src, dst := packVec4(in), make([]any, n*n)
	for i := range dst {
		dst[i] = []float64{0, 0, 0, 0}
	}
	log2n := bits.TrailingZeros(uint(n))
	for axis := int64(0); axis < 2; axis++ {
		for stage := 0; stage < log2n; stage++ {
			mem := &run.Memory{Vars: map[string]any{
				"fft": map[string]any{"n": int64(n), "stage": int64(stage), "axis": axis, "_pad": int64(0)},
				"src": src, "dst": dst,
			}}
			if err := run.RunWith(m, "fft_stage", n*n, mem, run.Options{Float32: true}); err != nil {
				t.Fatalf("axis=%d stage=%d: %v", axis, stage, err)
			}
			src, dst = dst, src
		}
	}
	got := unpackVec4(src)
	want := dft2DInverse(in, n)
	e, rms := maxErr(got, want), rmsOf(want)
	t.Logf("float32 gather FFT n=%d: max error %.3g, output RMS %.3g, relative %.3g", n, e, rms, e/rms)
	if math.IsNaN(e) || e > 1e-4*rms {
		t.Errorf("max error %g exceeds 1e-4 of the output RMS %g", e, rms)
	}
}
