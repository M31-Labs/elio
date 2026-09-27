package stdlib

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestGPUDrivenWGSLMatchesGoSX keeps the WGSL that GoSX embeds for its
// GPU-driven instancing host byte-identical to the goldens here. GoSX keeps
// copies in client/js/testdata/elio/ and checks its embedded strings against
// them, so this test closes the loop across the two repositories. It skips
// when the gosx sibling checkout (the go.mod replace target) is absent.
func TestGPUDrivenWGSLMatchesGoSX(t *testing.T) {
	pairs := []struct{ golden, gosx string }{
		{"gpudriven/cull.wgsl", "gpudriven_cull.wgsl"},
		{"gpudriven/hiz_downsample.wgsl", "gpudriven_hiz_downsample.wgsl"},
	}
	for _, pair := range pairs {
		want, err := os.ReadFile(pair.golden)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join("..", "..", "gosx", "client", "js", "testdata", "elio", pair.gosx)
		got, err := os.ReadFile(path)
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("gosx sibling checkout has no %s; run gosx task G05 first", path)
		}
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s differs from %s: regenerate the golden here, copy it to gosx, and re-embed it (spec task G05)", path, pair.golden)
		}
	}
}
