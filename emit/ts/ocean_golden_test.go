package ts_test

import (
	"os"
	"testing"

	"m31labs.dev/elio/emit/ts"
	"m31labs.dev/elio/kernelsets"
)

func TestOceanGolden(t *testing.T) {
	set, err := kernelsets.OceanWithGraphs()
	if err != nil {
		t.Fatal(err)
	}
	got, err := ts.Emit(set, ts.Meta{Version: "golden"})
	if err != nil {
		t.Fatal(err)
	}
	const path = "testdata/ocean.ts"
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(want) != got {
		t.Errorf("%s differs from emitted output; rerun with UPDATE_GOLDEN=1", path)
	}
}

func TestOceanGlobalGolden(t *testing.T) {
	set, err := kernelsets.OceanWithGraphs()
	if err != nil {
		t.Fatal(err)
	}
	got, err := ts.Emit(set, ts.Meta{Version: "golden", Global: "__gosx_scene3d_elio_ocean"})
	if err != nil {
		t.Fatal(err)
	}
	const path = "testdata/ocean.global.js"
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(want) != got {
		t.Errorf("%s differs from emitted output; rerun with UPDATE_GOLDEN=1", path)
	}
}

// TestOceanBackendGoldens pins the per-backend modules a page loads.
func TestOceanBackendGoldens(t *testing.T) {
	set, err := kernelsets.OceanWithGraphs()
	if err != nil {
		t.Fatal(err)
	}
	for _, backend := range []string{"webgpu", "webgl2"} {
		t.Run(backend, func(t *testing.T) {
			got, err := ts.Emit(set, ts.Meta{Version: "golden", Global: "__gosx_scene3d_elio_ocean", Backend: backend})
			if err != nil {
				t.Fatal(err)
			}
			path := "testdata/ocean." + backend + ".global.js"
			if os.Getenv("UPDATE_GOLDEN") != "" {
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if string(want) != got {
				t.Errorf("%s differs from emitted output; rerun with UPDATE_GOLDEN=1", path)
			}
		})
	}
}
