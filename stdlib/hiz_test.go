package stdlib

import (
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func TestHiZIsValid(t *testing.T) {
	if errs := sema.Check(HiZDownsample()); len(errs) != 0 {
		t.Fatalf("HiZDownsample failed sema:\n%v", sema.Errors(errs))
	}
}

// TestHiZDownsampleMaxes downsamples a 4×4 depth buffer (values 0..15) to 2×2,
// each destination texel the max of its 2×2 source block, and checks the result
// — the per-texel index math and the 4-way max executed on the CPU fallback.
func TestHiZDownsampleMaxes(t *testing.T) {
	// 4x4 source, row-major 0..15.
	src := make([]float64, 16)
	for i := range src {
		src[i] = float64(i)
	}
	dst := make([]float64, 4) // 2x2
	dims := map[string]any{"srcWidth": int64(4), "dstWidth": int64(2)}
	mem := &run.Memory{Vars: map[string]any{"dims": dims, "src": src, "dst": dst}}

	if err := run.Run(HiZDownsample(), "hiz", len(dst), mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	// max of each 2x2 block:
	//   {0,1,4,5}=5  {2,3,6,7}=7  {8,9,12,13}=13  {10,11,14,15}=15
	want := []float64{5, 7, 13, 15}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst = %v, want %v (mismatch at [%d])", dst, want, i)
		}
	}
}
