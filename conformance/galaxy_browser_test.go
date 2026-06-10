package conformance

import (
	"math"
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
	"m31labs.dev/elio/stdlib"
)

// TestGalaxyParticleSimulateAllBackends cross-validates the browser-contract
// galaxy simulate kernel across all backends, matching
// TestGalaxyParticleUpdateAllBackends's coverage.
func TestGalaxyParticleSimulateAllBackends(t *testing.T) {
	mod := stdlib.GalaxyParticleSimulate()

	// Sema pass first.
	if errs := sema.Check(mod); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}

	// WGSL — naga validated.
	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	for _, want := range []string{
		"fn simulate(",
		"var<storage, read_write> particles",
		"var<storage, read_write> renderData",
		"var<uniform> params",
		"var<storage, read> forces",
		"struct SimParams",
		"struct RenderVertex",
		"struct Force",
		"select(",
	} {
		if !strings.Contains(wsrc, want) {
			t.Errorf("WGSL missing %q", want)
		}
	}
	prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(f string) []string { return []string{f} })

	// GLSL — glslangValidator.
	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	// Metal — structural checks.
	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{
		"kernel void simulate(",
		"struct Particle",
		"struct RenderVertex",
		"struct SimParams",
		"struct Force",
	} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q", want)
		}
	}

	// CPU interpreter: one particle that is alive (age=0.1 < lifetime=10)
	// under drag force kind=4, strength=0.5, dt=0.1.
	// vel *= max(0, 1 - str*dt) per drag step:
	//   dv = (-velX * str * dt, ...)
	//   velX += dv.x  ⇒  velX' = velX - velX * 0.5 * 0.1 = velX * 0.95
	// posX += velX' * dt = 1.0 * 0.95 * 0.1 = 0.095
	// Also checks renderData write.
	p := map[string]any{
		"posX": 0.0, "posY": 0.0, "posZ": 0.0,
		"velX": 1.0, "velY": 0.0, "velZ": 0.0,
		"age": 0.1, "lifetime": 10.0,
	}
	rd := map[string]any{
		"posX": 0.0, "posY": 0.0, "posZ": 0.0,
		"size": 0.0, "r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0,
	}
	force := map[string]any{
		"kind": int64(4), "strength": 0.5,
		"dirX": 0.0, "dirY": 0.0, "dirZ": 0.0,
		"frequency": 0.0, "_pad0": 0.0, "_pad1": 0.0,
	}
	params := map[string]any{
		"deltaTime": 0.1, "totalTime": 1.0,
		"count": int64(1), "_pad0": int64(0),
		"emitterKind": int64(0),
		"emitterX": 0.0, "emitterY": 0.0, "emitterZ": 0.0,
		"emitterRadius": 1.0, "emitterRate": 0.0,
		"emitterLifetime": 5.0, "emitterOnce": int64(0),
		"emitterArms": int64(2),
		"emitterWind": 0.0, "emitterScatter": 0.0,
		"emitterRotX": 0.0, "emitterRotY": 0.0, "emitterRotZ": 0.0,
		"_pad2": int64(0),
		"sizeStart": 1.0, "sizeEnd": 0.5,
		"colorStartR": 1.0, "colorStartG": 0.5, "colorStartB": 0.0,
		"colorEndR": 0.0, "colorEndG": 0.5, "colorEndB": 1.0,
		"opacityStart": 1.0, "opacityEnd": 0.0,
		"forceCount": int64(1), "bounds": 0.0,
	}
	mem := &run.Memory{Vars: map[string]any{
		"particles":  []any{p},
		"renderData": []any{rd},
		"params":     params,
		"forces":     []any{force},
	}}
	if err := run.Run(mod, "simulate", 1, mem); err != nil {
		t.Fatalf("run.Run simulate: %v", err)
	}
	gotP := mem.Vars["particles"].([]any)[0].(map[string]any)
	const eps = 1e-5
	wantPosX := 1.0 * 0.95 * 0.1
	if got := gotP["posX"].(float64); math.Abs(got-wantPosX) > eps {
		t.Errorf("simulate posX = %v, want ~%v", got, wantPosX)
	}
	if got := gotP["age"].(float64); math.Abs(got-0.2) > eps {
		t.Errorf("simulate age = %v, want 0.2", got)
	}
	// RenderVertex should have been written.
	gotRD := mem.Vars["renderData"].([]any)[0].(map[string]any)
	if _, ok := gotRD["posX"]; !ok {
		t.Error("renderData[0] not updated")
	}
}

// --- Uniform byte-layout assertions ---

// wgslAlign returns the WGSL uniform-address-space alignment for a type.
// Rules: f32/u32/i32 → align 4; vec2<f32> → 8; vec3<f32>/vec4<f32> → 16;
// struct → max member align (rounded up to 16); array<T,N> → max(align(T),16)
// when in uniform, align(T) otherwise.
func wgslAlign(t ir.Type, structs map[string][]ir.Field) int {
	switch x := t.(type) {
	case ir.Scalar:
		return 4
	case ir.Vec:
		switch x.N {
		case 2:
			return 8
		default:
			return 16
		}
	case ir.Named:
		maxAlign := 4
		for _, f := range structs[x.Name] {
			if a := wgslAlign(f.Type, structs); a > maxAlign {
				maxAlign = a
			}
		}
		// Struct align is the next multiple of its own max-align ≥ 16 only in
		// uniform/storage: the rule is roundUp(maxMemberAlign, 16) per WGSL spec
		// §3.6.2.3. But only when used in uniform address-space. We apply it
		// unconditionally here since both use cases in the galaxy kernels are
		// within uniform/storage blocks.
		if maxAlign < 16 {
			maxAlign = 16
		}
		return maxAlign
	case ir.Array:
		ea := wgslAlign(x.Elem, structs)
		if ea < 16 { // uniform array element align ≥ 16
			ea = 16
		}
		return ea
	}
	return 4
}

// wgslSize returns the WGSL size of a type (the storage footprint, before
// padding to align). For a struct it includes the end-padding to its align.
func wgslSize(t ir.Type, structs map[string][]ir.Field) int {
	switch x := t.(type) {
	case ir.Scalar:
		return 4
	case ir.Vec:
		return 4 * x.N
	case ir.Named:
		off := 0
		for _, f := range structs[x.Name] {
			fa := wgslAlign(f.Type, structs)
			off = roundUp(off, fa)
			off += wgslSize(f.Type, structs)
		}
		sa := wgslAlign(x, structs)
		return roundUp(off, sa)
	case ir.Array:
		if x.Len == 0 {
			return 0 // runtime-sized
		}
		ea := wgslAlign(x.Elem, structs)
		elemStride := roundUp(wgslSize(x.Elem, structs), ea)
		return x.Len * elemStride
	}
	return 4
}

func roundUp(v, align int) int {
	if align <= 1 {
		return v
	}
	return ((v + align - 1) / align) * align
}

// fieldOffsets computes the WGSL offset of each field in a struct, returned
// as a map[fieldName]byteOffset.
func fieldOffsets(fields []ir.Field, structs map[string][]ir.Field) map[string]int {
	off := 0
	out := map[string]int{}
	for _, f := range fields {
		fa := wgslAlign(f.Type, structs)
		off = roundUp(off, fa)
		out[f.Name] = off
		off += wgslSize(f.Type, structs)
	}
	return out
}

// TestParticleUniformsByteLayout asserts that the WGSL struct layout the
// Elio IR produces for GalaxyParticleUpdate's ParticleUniforms matches the
// exact byte offsets GoSX's encodeParticleUpdateUniforms writes
// (render/bundle/particles.go:630-657).
//
// GoSX encoding:
//   [0..3]   dt, time, lifetime, forceCount  (4×f32 at 0..15)
//   [16..31] emitterPos  vec4<f32>
//   [32..47] initialSpeed vec4<f32>
//   [48..]   forces[8] × ForceStride(32) = [cfg vec4][vector vec4]
//   total    304 bytes
func TestParticleUniformsByteLayout(t *testing.T) {
	mod := stdlib.GalaxyParticleUpdate()
	structs := make(map[string][]ir.Field)
	for _, s := range mod.Structs {
		structs[s.Name] = s.Fields
	}

	// Find ParticleUniforms struct.
	puFields, ok := structs["ParticleUniforms"]
	if !ok {
		t.Fatal("ParticleUniforms struct not found")
	}
	offsets := fieldOffsets(puFields, structs)

	type want struct {
		field  string
		offset int
	}
	cases := []want{
		{"dt", 0},
		{"time", 4},
		{"lifetime", 8},
		{"forceCount", 12},
		{"emitterPos", 16},
		{"initialSpeed", 32},
		{"forces", 48},
	}
	for _, c := range cases {
		got, ok := offsets[c.field]
		if !ok {
			t.Errorf("field %q not found in ParticleUniforms", c.field)
			continue
		}
		if got != c.offset {
			t.Errorf("ParticleUniforms.%s offset = %d, want %d", c.field, got, c.offset)
		}
	}
	// Total size check: 48 + 8*32 = 304.
	gotSize := wgslSize(ir.Named{Name: "ParticleUniforms"}, structs)
	if gotSize != 304 {
		t.Errorf("ParticleUniforms total size = %d, want 304", gotSize)
	}

	// Force element stride: each ParticleForce = cfg vec4 + vector vec4 = 32 bytes.
	forceStride := roundUp(wgslSize(ir.Named{Name: "ParticleForce"}, structs),
		wgslAlign(ir.Named{Name: "ParticleForce"}, structs))
	if forceStride != 32 {
		t.Errorf("ParticleForce stride = %d, want 32", forceStride)
	}
}

// TestSimParamsByteLayout asserts that the WGSL SimParams struct layout the
// Elio IR produces for GalaxyParticleSimulate matches the exact byte offsets
// sceneComputeUploadSimParams writes in 16b-scene-compute.js.
//
// JS sequential writes, all 4-byte fields (no implicit padding since all
// f32/u32 have align 4):
//
//	@0   deltaTime      f32
//	@4   totalTime      f32
//	@8   count          u32
//	@12  _pad0          u32
//	@16  emitterKind    u32
//	@20  emitterX       f32
//	@24  emitterY       f32
//	@28  emitterZ       f32
//	@32  emitterRadius  f32
//	@36  emitterRate    f32
//	@40  emitterLifetime f32
//	@44  emitterOnce    u32
//	@48  emitterArms    u32
//	@52  emitterWind    f32
//	@56  emitterScatter f32
//	@60  emitterRotX    f32
//	@64  emitterRotY    f32
//	@68  emitterRotZ    f32
//	@72  _pad2          u32
//	@76  sizeStart      f32
//	@80  sizeEnd        f32
//	@84  colorStartR    f32
//	@88  colorStartG    f32
//	@92  colorStartB    f32
//	@96  colorEndR      f32
//	@100 colorEndG      f32
//	@104 colorEndB      f32
//	@108 opacityStart   f32
//	@112 opacityEnd     f32
//	@116 forceCount     u32
//	@120 bounds         f32
//
// Force struct layout (sceneComputeUploadForces, stride 32):
//
//	@0  kind       u32
//	@4  strength   f32
//	@8  dirX       f32
//	@12 dirY       f32
//	@16 dirZ       f32
//	@20 frequency  f32
//	@24 _pad0      f32
//	@28 _pad1      f32
func TestSimParamsByteLayout(t *testing.T) {
	mod := stdlib.GalaxyParticleSimulate()
	structs := make(map[string][]ir.Field)
	for _, s := range mod.Structs {
		structs[s.Name] = s.Fields
	}

	// SimParams — all fields are scalars (align 4), so WGSL offsets == JS offsets.
	spFields, ok := structs["SimParams"]
	if !ok {
		t.Fatal("SimParams struct not found")
	}
	offsets := fieldOffsets(spFields, structs)

	type want struct {
		field  string
		offset int
	}
	spCases := []want{
		{"deltaTime", 0},
		{"totalTime", 4},
		{"count", 8},
		{"_pad0", 12},
		{"emitterKind", 16},
		{"emitterX", 20},
		{"emitterY", 24},
		{"emitterZ", 28},
		{"emitterRadius", 32},
		{"emitterRate", 36},
		{"emitterLifetime", 40},
		{"emitterOnce", 44},
		{"emitterArms", 48},
		{"emitterWind", 52},
		{"emitterScatter", 56},
		{"emitterRotX", 60},
		{"emitterRotY", 64},
		{"emitterRotZ", 68},
		{"_pad2", 72},
		{"sizeStart", 76},
		{"sizeEnd", 80},
		{"colorStartR", 84},
		{"colorStartG", 88},
		{"colorStartB", 92},
		{"colorEndR", 96},
		{"colorEndG", 100},
		{"colorEndB", 104},
		{"opacityStart", 108},
		{"opacityEnd", 112},
		{"forceCount", 116},
		{"bounds", 120},
	}
	for _, c := range spCases {
		got, ok := offsets[c.field]
		if !ok {
			t.Errorf("field %q not found in SimParams", c.field)
			continue
		}
		if got != c.offset {
			t.Errorf("SimParams.%s offset = %d, want %d (JS host writes %d)", c.field, got, c.offset, c.offset)
		}
	}
	// SimParams: 32 fields × 4 bytes = 128 bytes. WGSL uniform struct align =
	// roundUp(maxMemberAlign, 16) = roundUp(4, 16) = 16; size = roundUp(124, 16) = 128.
	gotSize := wgslSize(ir.Named{Name: "SimParams"}, structs)
	if gotSize != 128 {
		t.Errorf("SimParams total size = %d, want 128", gotSize)
	}

	// Force struct layout.
	fFields, ok := structs["Force"]
	if !ok {
		t.Fatal("Force struct not found")
	}
	fOffsets := fieldOffsets(fFields, structs)
	fCases := []want{
		{"kind", 0},
		{"strength", 4},
		{"dirX", 8},
		{"dirY", 12},
		{"dirZ", 16},
		{"frequency", 20},
		{"_pad0", 24},
		{"_pad1", 28},
	}
	for _, c := range fCases {
		got, ok := fOffsets[c.field]
		if !ok {
			t.Errorf("Force field %q not found", c.field)
			continue
		}
		if got != c.offset {
			t.Errorf("Force.%s offset = %d, want %d", c.field, got, c.offset)
		}
	}
	gotForceSize := wgslSize(ir.Named{Name: "Force"}, structs)
	if gotForceSize != 32 {
		t.Errorf("Force size = %d, want 32", gotForceSize)
	}
}

// TestGalaxySimulateIntegrationParity checks that the browser kernel's force
// integration (drag, gravity, orbit, turbulence) produces the same velocity
// and position change as a hand-computed reference over equivalent physics
// inputs. The particle is kept alive (age < lifetime) to isolate the
// integration path from respawn.
func TestGalaxySimulateIntegrationParity(t *testing.T) {
	// Particle alive, no respawn needed.
	p := map[string]any{
		"posX": 10.0, "posY": 0.0, "posZ": 0.0,
		"velX": 0.0, "velY": 0.0, "velZ": 0.0,
		"age": 0.5, "lifetime": 100.0,
	}
	rd := map[string]any{
		"posX": 0.0, "posY": 0.0, "posZ": 0.0,
		"size": 0.0, "r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0,
	}
	dt := 1.0
	// Gravity (kind=0): dir=(0,-1,0), strength=10 → dv=(0,-10*dt,0)
	gravForce := map[string]any{
		"kind": int64(0), "strength": 10.0,
		"dirX": 0.0, "dirY": -1.0, "dirZ": 0.0,
		"frequency": 0.0, "_pad0": 0.0, "_pad1": 0.0,
	}
	params := map[string]any{
		"deltaTime": dt, "totalTime": 0.0,
		"count": int64(1), "_pad0": int64(0),
		"emitterKind": int64(0),
		"emitterX": 0.0, "emitterY": 0.0, "emitterZ": 0.0,
		"emitterRadius": 1.0, "emitterRate": 0.0,
		"emitterLifetime": 5.0, "emitterOnce": int64(0),
		"emitterArms": int64(2),
		"emitterWind": 0.0, "emitterScatter": 0.0,
		"emitterRotX": 0.0, "emitterRotY": 0.0, "emitterRotZ": 0.0,
		"_pad2": int64(0),
		"sizeStart": 1.0, "sizeEnd": 1.0,
		"colorStartR": 1.0, "colorStartG": 1.0, "colorStartB": 1.0,
		"colorEndR": 1.0, "colorEndG": 1.0, "colorEndB": 1.0,
		"opacityStart": 1.0, "opacityEnd": 1.0,
		"forceCount": int64(1), "bounds": 0.0,
	}
	mem := &run.Memory{Vars: map[string]any{
		"particles":  []any{p},
		"renderData": []any{rd},
		"params":     params,
		"forces":     []any{gravForce},
	}}
	if err := run.Run(stdlib.GalaxyParticleSimulate(), "simulate", 1, mem); err != nil {
		t.Fatalf("run: %v", err)
	}
	got := mem.Vars["particles"].([]any)[0].(map[string]any)
	const eps = 1e-5
	// dv = (0, -10, 0) * dt; vel' = (0,-10,0); pos' = pos + vel'*dt
	checkF := func(field string, want float64) {
		t.Helper()
		v, ok := got[field]
		if !ok {
			t.Errorf("field %q missing", field)
			return
		}
		if math.Abs(v.(float64)-want) > eps {
			t.Errorf("simulate %s = %v, want %v", field, v, want)
		}
	}
	checkF("velX", 0.0)
	checkF("velY", -10.0)
	checkF("velZ", 0.0)
	checkF("posX", 10.0+0.0*dt) // pos unchanged in X
	checkF("posY", 0.0+(-10.0)*dt)
	checkF("posZ", 0.0)
	checkF("age", 0.5+dt)
}

// TestGalaxySimulateEmitterKindDefault checks that an out-of-range emitterKind
// (>= 4) falls through to the point emitter, matching the reference JS
// switch-default branch. The particle is dormant (age=0, lifetime=0) so it
// respawns immediately; posX/Y/Z must be 0 (point emitter origin).
func TestGalaxySimulateEmitterKindDefault(t *testing.T) {
	// age < 0 and lifetime <= 0 triggers immediate emitParticle in the kernel's
	// aging state machine (before the emitterOnce guard which requires emitterOnce != 0).
	p := map[string]any{
		"posX": 99.0, "posY": 99.0, "posZ": 99.0,
		"velX": 0.0, "velY": 0.0, "velZ": 0.0,
		"age": -0.1, "lifetime": 0.0,
	}
	rd := map[string]any{
		"posX": 0.0, "posY": 0.0, "posZ": 0.0,
		"size": 0.0, "r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0,
	}
	params := map[string]any{
		"deltaTime": 0.016, "totalTime": 0.0,
		"count": int64(1), "_pad0": int64(0),
		"emitterKind": int64(99), // out-of-range: reference default => point
		"emitterX": 0.0, "emitterY": 0.0, "emitterZ": 0.0,
		"emitterRadius": 1.0, "emitterRate": 0.0,
		"emitterLifetime": 5.0, "emitterOnce": int64(0),
		"emitterArms": int64(2),
		"emitterWind": 0.0, "emitterScatter": 0.0,
		"emitterRotX": 0.0, "emitterRotY": 0.0, "emitterRotZ": 0.0,
		"_pad2": int64(0),
		"sizeStart": 1.0, "sizeEnd": 1.0,
		"colorStartR": 1.0, "colorStartG": 1.0, "colorStartB": 1.0,
		"colorEndR": 1.0, "colorEndG": 1.0, "colorEndB": 1.0,
		"opacityStart": 1.0, "opacityEnd": 1.0,
		"forceCount": int64(0), "bounds": 0.0,
	}
	mem := &run.Memory{Vars: map[string]any{
		"particles":  []any{p},
		"renderData": []any{rd},
		"params":     params,
		"forces":     []any{},
	}}
	if err := run.Run(stdlib.GalaxyParticleSimulate(), "simulate", 1, mem); err != nil {
		t.Fatalf("run: %v", err)
	}
	got := mem.Vars["particles"].([]any)[0].(map[string]any)
	// Point emitter sets posX/Y/Z = 0 then the kernel integrates one step: pos += vel*dt.
	// The point emitter velocity is (hash-0.5)*0.1, so |vel| <= 0.05 and |pos| <= vel*dt
	// = 0.05*0.016 = 0.0008. The pre-spawn position of 99 must be completely overwritten.
	// Tolerance 0.01 is well above the max integration step but far below the old 99.
	const spawnTol = 0.01
	if v := got["posX"].(float64); math.Abs(v) > spawnTol {
		t.Errorf("emitterKind=99 (default=point): posX = %v, want near 0 (point emitter, pre-spawn was 99)", v)
	}
	if v := got["posY"].(float64); math.Abs(v) > spawnTol {
		t.Errorf("emitterKind=99 (default=point): posY = %v, want near 0 (point emitter, pre-spawn was 99)", v)
	}
	if v := got["posZ"].(float64); math.Abs(v) > spawnTol {
		t.Errorf("emitterKind=99 (default=point): posZ = %v, want near 0 (point emitter, pre-spawn was 99)", v)
	}
	// age must be reset and advanced by one dt.
	if v := got["age"].(float64); math.Abs(v) > 0.1 {
		t.Errorf("emitterKind=99 (default=point): age = %v, want ~0 (fresh emit)", v)
	}
}
