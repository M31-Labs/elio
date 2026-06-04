package main

import (
	"bytes"
	"encoding/json"
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

func TestCheckRendersSourceDiagnostic(t *testing.T) {
	f := filepath.Join(t.TempDir(), "bad.elio")
	src := `@workgroup(1) kernel main() {
  let x = missing;
}
`
	if err := os.WriteFile(f, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := run([]string{"check", f}, &out, &errb); code != 1 {
		t.Fatalf("check: want exit 1, got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	for _, want := range []string{"ELIO1001 error", "--> " + f + ":2:", "let x = missing;", "^", "hint:"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("diagnostic output missing %q\n--- stderr ---\n%s", want, errb.String())
		}
	}
}

func TestGraphPrintsJSON(t *testing.T) {
	f := writeSample(t)
	var out, errb bytes.Buffer
	if code := run([]string{"graph", "--format", "json", f}, &out, &errb); code != 0 {
		t.Fatalf("graph: exit %d: %s", code, errb.String())
	}
	var payload struct {
		GraphVersion int `json:"graph_version"`
		Counts       struct {
			Kernels  int `json:"kernels"`
			Bindings int `json:"bindings"`
		} `json:"counts"`
		Module struct {
			Kernels []struct {
				Name       string `json:"name"`
				Statements int    `json:"statements"`
			} `json:"kernels"`
		} `json:"module"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("unmarshal graph: %v\n%s", err, out.String())
	}
	if payload.GraphVersion != 1 || payload.Counts.Kernels != 1 || payload.Counts.Bindings != 1 {
		t.Fatalf("unexpected graph payload: %+v", payload)
	}
	if len(payload.Module.Kernels) != 1 || payload.Module.Kernels[0].Name != "main" || payload.Module.Kernels[0].Statements == 0 {
		t.Fatalf("missing kernel snapshot: %+v", payload.Module.Kernels)
	}
}

func TestKernelsExtractsTargetSourcesWithValidation(t *testing.T) {
	f := writeSample(t)
	outDir := filepath.Join(t.TempDir(), "kernels")
	var out, errb bytes.Buffer
	if code := run([]string{"kernels", "--target", "wgsl", "--validate", "--out", outDir, f}, &out, &errb); code != 0 {
		t.Fatalf("kernels: exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "wrote 1 target source") {
		t.Fatalf("unexpected kernels output: %s", out.String())
	}
	data, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest struct {
		SourceCount int `json:"source_count"`
		Sources     []struct {
			Target     string `json:"target"`
			SourceFile string `json:"source_file"`
			Kernels    []struct {
				Entry        string `json:"entry"`
				EntryChecked bool   `json:"entry_checked"`
			} `json:"kernels"`
		} `json:"sources"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal manifest: %v\n%s", err, data)
	}
	if manifest.SourceCount != 1 || len(manifest.Sources) != 1 || manifest.Sources[0].Target != "wgsl" {
		t.Fatalf("unexpected manifest: %+v", manifest)
	}
	if len(manifest.Sources[0].Kernels) != 1 || !manifest.Sources[0].Kernels[0].EntryChecked || manifest.Sources[0].Kernels[0].Entry != "main" {
		t.Fatalf("missing validation metadata: %+v", manifest.Sources[0].Kernels)
	}
	src, err := os.ReadFile(filepath.Join(outDir, manifest.Sources[0].SourceFile))
	if err != nil {
		t.Fatalf("read source: %v", err)
	}
	if !strings.Contains(string(src), "@compute") {
		t.Fatalf("extracted WGSL missing compute entry:\n%s", src)
	}
}

func TestCompileBundleWritesInspectionArtifacts(t *testing.T) {
	f := writeSample(t)
	bundleDir := filepath.Join(t.TempDir(), "bundle")
	var out, errb bytes.Buffer
	if code := run([]string{"compile", "--bundle", bundleDir, "--validate-kernels", f}, &out, &errb); code != 0 {
		t.Fatalf("compile bundle: exit %d: %s", code, errb.String())
	}
	for _, path := range []string{
		filepath.Join(bundleDir, "manifest.json"),
		filepath.Join(bundleDir, "source.elio"),
		filepath.Join(bundleDir, "graph.json"),
		filepath.Join(bundleDir, "kernels"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected bundle path %q: %v", path, err)
		}
	}
}

func TestDoctorReportsTools(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"doctor"}, &out, &errb); code != 0 {
		t.Fatalf("doctor: exit %d: %s", code, errb.String())
	}
	for _, want := range []string{"elio: dev", "targets:", "tools:", "naga"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("doctor output missing %q\n%s", want, out.String())
		}
	}
}
