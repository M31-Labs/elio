package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const sample = `@group(0) @binding(0) storage read_write data: []f32;

@workgroup(64) kernel main(gid: global_invocation_id) {
  let i = gid.x;
  if i >= arrayLength(&data) { return; }
}
`

func writeSample(t *testing.T) string {
	t.Helper()
	f := filepath.Join(t.TempDir(), "k.elio")
	if err := os.WriteFile(f, []byte(sample), 0o644); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestEmitEachTarget(t *testing.T) {
	f := writeSample(t)
	for _, tgt := range []string{"wgsl", "glsl", "metal"} {
		var out, errb bytes.Buffer
		if code := run([]string{"emit", tgt, f}, &out, &errb); code != 0 {
			t.Fatalf("emit %s: exit %d: %s", tgt, code, errb.String())
		}
		if out.Len() == 0 {
			t.Errorf("emit %s produced no output", tgt)
		}
		if !strings.Contains(out.String(), "main") {
			t.Errorf("emit %s missing entrypoint:\n%s", tgt, out.String())
		}
	}
}

func TestCheckAndErrors(t *testing.T) {
	f := writeSample(t)

	var out, errb bytes.Buffer
	if code := run([]string{"check", f}, &out, &errb); code != 0 {
		t.Fatalf("check: exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "ok") {
		t.Errorf("check: want ok, got %q", out.String())
	}

	if code := run([]string{"emit", "spirv", f}, io.Discard, io.Discard); code != 2 {
		t.Errorf("unknown target: want exit 2, got %d", code)
	}
	if code := run([]string{"emit", "wgsl", "/no/such.elio"}, io.Discard, io.Discard); code != 1 {
		t.Errorf("missing file: want exit 1, got %d", code)
	}
	if code := run(nil, io.Discard, io.Discard); code != 2 {
		t.Errorf("no args: want exit 2, got %d", code)
	}
}
