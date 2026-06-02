package conformance

import (
	"strings"
	"testing"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
)

// TestSkinLBSAllBackends proves the linear-blend skinning kernel emits valid
// shaders on every text backend: naga validates the WGSL, glslangValidator
// compiles the GLSL to SPIR-V, and the Metal output is checked structurally.
// (CPU execution of this kernel lives in run.TestRunSkinLBS.)
func TestSkinLBSAllBackends(t *testing.T) {
	mod := ir.SkinLBS()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "skin.wgsl", wsrc, func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "skin.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"kernel void main(", "outPos"} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}
}

// TestSkinDQSAllBackends proves the dual-quaternion skinning kernel — the most
// intricate of the skinning kernels, exercising dot, the sqrt builtin, cross
// products spelled out as scalar arithmetic, and componentwise member-assign to
// dst[i].{x,y,z,w} — emits valid shaders on every text backend: naga validates
// the WGSL, glslangValidator compiles the GLSL to SPIR-V, and the Metal output
// is checked structurally. (CPU execution of this kernel lives in
// run.TestRunSkinDQS, the correctness anchor.)
func TestSkinDQSAllBackends(t *testing.T) {
	mod := ir.SkinDQS()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "dqs.wgsl", wsrc, func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "dqs.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"kernel void main(", "dst"} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}
}

// TestSqrtAllBackends validates the sqrt builtin emits on every backend.
func TestSqrtAllBackends(t *testing.T) {
	mod := ir.SqrtKernel()
	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "sqrt.wgsl", wsrc, func(f string) []string { return []string{f} })
	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "sqrt.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })
	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"kernel void main(", "sqrt("} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}
}
