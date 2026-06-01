// Package conformance cross-checks a single Elio kernel across every backend —
// WGSL (naga), GLSL (glslang→SPIR-V), Metal (structural), and the CPU
// interpreter (executed) — so the IR, emitters, and interpreter are proven to
// agree on more than the cull kernel.
package conformance

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/run"
)

// TestScaleBiasAllBackends proves the second kernel (vector arithmetic over
// vec4 buffers) emits valid shaders on every text backend and computes the
// right result on the CPU fallback.
func TestScaleBiasAllBackends(t *testing.T) {
	mod := ir.ScaleBias()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "scale.wgsl", wsrc, func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "scale.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"kernel void main(", "device float4* dst", "dst[i] ="} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}

	// CPU execution: dst[i] = src[i]*scale + bias.
	src := []any{[]float64{1, 2, 3, 4}, []float64{5, 6, 7, 8}}
	dst := []any{nil, nil}
	mem := &run.Memory{Vars: map[string]any{
		"params": map[string]any{"scale": []float64{2, 2, 2, 2}, "bias": []float64{1, 1, 1, 1}},
		"src":    src,
		"dst":    dst,
	}}
	if err := run.Run(mod, "main", len(src), mem); err != nil {
		t.Fatalf("run.Run: %v", err)
	}
	if !eqVec(dst[0], []float64{3, 5, 7, 9}) {
		t.Errorf("dst[0] = %v, want [3 5 7 9]", dst[0])
	}
	if !eqVec(dst[1], []float64{11, 13, 15, 17}) {
		t.Errorf("dst[1] = %v, want [11 13 15 17]", dst[1])
	}
}

func validate(t *testing.T, bin, fname, src string, args func(string) []string) {
	t.Helper()
	path, err := exec.LookPath(bin)
	if err != nil {
		t.Logf("%s: skipped (not installed)", bin)
		return
	}
	dir := t.TempDir()
	f := filepath.Join(dir, fname)
	if err := os.WriteFile(f, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(path, args(f)...).CombinedOutput()
	if err != nil {
		t.Fatalf("%s validation failed: %v\n%s\n--- source ---\n%s", bin, err, out, src)
	}
	if strings.Contains(string(out), "ERROR") {
		t.Errorf("%s reported errors:\n%s", bin, out)
	}
}

func eqVec(got any, want []float64) bool {
	v, ok := got.([]float64)
	if !ok || len(v) != len(want) {
		return false
	}
	for i := range v {
		if v[i] != want[i] {
			return false
		}
	}
	return true
}
