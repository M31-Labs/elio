package stdlib

import (
	"testing"

	"m31labs.dev/elio/sema"
)

// TestGPUDrivenKernelsAreValid pins that both embedded GoSX kernels parse,
// pass semantic analysis, and expose the entry names GoSX dispatches.
func TestGPUDrivenKernelsAreValid(t *testing.T) {
	cull, err := GPUDrivenCull()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(cull); len(errs) != 0 {
		t.Fatalf("cull.elio failed sema:\n%v", sema.Errors(errs))
	}
	down, err := GPUDrivenHiZDownsample()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(down); len(errs) != 0 {
		t.Fatalf("hiz_downsample.elio failed sema:\n%v", sema.Errors(errs))
	}
	if got := cull.Kernels[0].Name; got != "cull" {
		t.Fatalf("cull entry = %q, want cull", got)
	}
	if got := down.Kernels[0].Name; got != "downsample" {
		t.Fatalf("downsample entry = %q, want downsample", got)
	}
}
