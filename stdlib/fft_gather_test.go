package stdlib

import (
	"math/bits"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func TestFFTSizesRejected(t *testing.T) {
	for _, n := range []int{-4, 0, 1, 2, 3, 6, 100, 512, 1024} {
		if _, err := FFT2DGather(n); err == nil {
			t.Errorf("FFT2DGather(%d): want error", n)
		}
		if _, err := FFT2DCompute(n); err == nil {
			t.Errorf("FFT2DCompute(%d): want error", n)
		}
	}
}

func TestFFT2DGatherMatchesDFT(t *testing.T) {
	for _, n := range []int{4, 8, 16, 64} {
		m, err := FFT2DGather(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		if errs := sema.Check(m); len(errs) != 0 {
			t.Fatalf("n=%d sema: %v", n, sema.Errors(errs))
		}
		in := randomSpectrum(n, int64(n))
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
				if err := run.Run(m, "fft_stage", n*n, mem); err != nil {
					t.Fatalf("n=%d axis=%d stage=%d: %v", n, axis, stage, err)
				}
				src, dst = dst, src
			}
		}
		got := unpackVec4(src)
		want := dft2DInverse(in, n)
		e, bound := maxErr(got, want), 1e-9*rmsOf(in)*float64(n)
		t.Logf("gather n=%d max error %.3g (bound %.3g)", n, e, bound)
		if e > bound {
			t.Errorf("n=%d: max error %g > %g", n, e, bound)
		}
	}
}
