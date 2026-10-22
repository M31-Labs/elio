package ts_test

import (
	"os"
	"testing"

	"m31labs.dev/elio/emit/ts"
	"m31labs.dev/elio/kernelsets"
)

func TestOceanGolden(t *testing.T) {
	set, err := kernelsets.Ocean()
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
