package ir

import "testing"

// TestSkinLBSModuleShape asserts the linear-blend skinning kernel has the
// expected coarse shape: one entrypoint named "main" and the five storage
// bindings of the buffer contract (restPos, joints, weights, palette, outPos).
func TestSkinLBSModuleShape(t *testing.T) {
	mod := SkinLBS()

	if got := len(mod.Kernels); got != 1 {
		t.Fatalf("kernels = %d, want 1", got)
	}
	if got := mod.Kernels[0].Name; got != "main" {
		t.Errorf("kernel name = %q, want %q", got, "main")
	}
	if got := len(mod.Bindings); got != 5 {
		t.Errorf("bindings = %d, want 5", got)
	}
}
