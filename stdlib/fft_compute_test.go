package stdlib

import (
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func TestFFT2DComputeMatchesDFT(t *testing.T) {
	for _, n := range []int{4, 8, 16, 64} {
		m, err := FFT2DCompute(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		if errs := sema.Check(m); len(errs) != 0 {
			t.Fatalf("n=%d sema: %v", n, sema.Errors(errs))
		}
		for _, k := range m.Kernels {
			if k.WorkgroupSize != [3]int{n / 2, 1, 1} {
				t.Errorf("n=%d %s workgroup size %v, want [%d 1 1]", n, k.Name, k.WorkgroupSize, n/2)
			}
		}
		in := randomSpectrum(n, int64(100+n))
		data := packVec4(in)
		mem := &run.Memory{Vars: map[string]any{"data": data}}
		for _, kernel := range []string{"fft_rows", "fft_cols"} {
			// n workgroups of n/2 lanes.
			if err := run.Run(m, kernel, n*(n/2), mem); err != nil {
				t.Fatalf("n=%d %s: %v", n, kernel, err)
			}
		}
		got := unpackVec4(data)
		want := dft2DInverse(in, n)
		e, bound := maxErr(got, want), 1e-9*rmsOf(in)*float64(n)
		t.Logf("compute n=%d max error %.3g (bound %.3g)", n, e, bound)
		if e > bound {
			t.Errorf("n=%d: max error %g > %g", n, e, bound)
		}
	}
}
