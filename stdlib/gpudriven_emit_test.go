package stdlib

import (
	"os"
	"path/filepath"
	"testing"

	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
)

// TestGPUDrivenWGSLGoldens pins the exact WGSL GoSX embeds. GoSX copies these
// two files byte for byte into client/js/testdata/elio/ and inlines the same
// text in client/runtime/scene3d/indirect-instancing.ts, so a change here is a change
// to the browser runtime. Regenerate with:
//
//	UPDATE_GOLDEN=1 go test ./stdlib -run TestGPUDrivenWGSLGoldens
func TestGPUDrivenWGSLGoldens(t *testing.T) {
	for _, tc := range []struct {
		golden string
		build  func() (*ir.Module, error)
	}{
		{"cull.wgsl", GPUDrivenCull},
		{"hiz_downsample.wgsl", GPUDrivenHiZDownsample},
	} {
		mod, err := tc.build()
		if err != nil {
			t.Fatal(err)
		}
		src, err := wgsl.Emit(mod)
		if err != nil {
			t.Fatalf("%s: emit: %v", tc.golden, err)
		}
		path := filepath.Join("gpudriven", tc.golden)
		if os.Getenv("UPDATE_GOLDEN") != "" {
			if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s (run UPDATE_GOLDEN=1 to create): %v", path, err)
		}
		if src != string(want) {
			t.Errorf("%s: emitted WGSL differs from the golden; if the change is intended, regenerate it and re-sync GoSX", tc.golden)
		}
	}
}
