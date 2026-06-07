package conformance

import (
	"math"
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/run"
	"m31labs.dev/elio/stdlib"
)

// TestGalaxyParticleUpdateAllBackends cross-validates the galaxy particle/star
// integrator — the single-source Elio replacement for GoSX's hand-written
// particleUpdateWGSL — across every backend. It exercises the full surface that
// makes the replacement possible: vector constructors (vec3/vec4), a scalar
// cast (f32) in the dynamic-count force loop, the math builtins
// (normalize/cross/dot/sin/cos/fract/abs/clamp), a uniform array of structs,
// and the integrate/respawn branches. WGSL is naga-validated, GLSL is compiled
// to SPIR-V by glslang, Metal is structurally checked, and the CPU interpreter
// executes the gravity-integration path.
func TestGalaxyParticleUpdateAllBackends(t *testing.T) {
	mod := stdlib.GalaxyParticleUpdate()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	for _, want := range []string{"vec3<f32>(", "vec4<f32>(", "f32(fi)", "normalize(cross(", "array<ParticleForce, 8>"} {
		if !strings.Contains(wsrc, want) {
			t.Errorf("WGSL missing %q\n%s", want, wsrc)
		}
	}
	prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"kernel void update(", "float3(", "float4(", "normalize(cross("} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}

	// CPU execution: one particle under gravity (strength 10, down), dt 1.
	// v += accel*dt ⇒ (2,-10,0); p += v*dt ⇒ (2,-10,0); age ⇒ 1.
	p := map[string]any{"position": []float64{0, 0, 0, 0}, "velocity": []float64{2, 0, 0, 10}}
	u := map[string]any{
		"dt": 1.0, "time": 0.0, "lifetime": 10.0, "forceCount": 1.0,
		"emitterPos": []float64{0, 0, 0, 0}, "initialSpeed": []float64{0, 0, 0, 0},
		"forces": []any{map[string]any{"cfg": []float64{1, 10, 0, 0}, "vector": []float64{0, -1, 0, 0}}},
	}
	mem := &run.Memory{Vars: map[string]any{"u": u, "particles": []any{p}}}
	if err := run.Run(mod, "update", 1, mem); err != nil {
		t.Fatalf("run.Run: %v", err)
	}
	pos := p["position"].([]float64)
	vel := p["velocity"].([]float64)
	want := []float64{2, -10, 0, 1}
	for i := range want {
		if math.Abs(pos[i]-want[i]) > 1e-6 {
			t.Errorf("position = %v, want %v", pos, want)
			break
		}
	}
	wantV := []float64{2, -10, 0, 10}
	for i := range wantV {
		if math.Abs(vel[i]-wantV[i]) > 1e-6 {
			t.Errorf("velocity = %v, want %v", vel, wantV)
			break
		}
	}
}
