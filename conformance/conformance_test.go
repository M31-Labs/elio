// Package conformance cross-checks Elio kernels across every backend — WGSL
// (naga), GLSL (glslang→SPIR-V), Metal (structural), and the CPU interpreter
// (executed) — so the IR, emitters, and interpreter are proven to agree. It
// covers the flagship kernels — cull (atomics, control flow, swizzles) and
// ScaleBias (vector arithmetic) — plus the cooperative stdlib primitives
// (reduce, scan) that use workgroup-shared memory and barriers.
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
	"m31labs.dev/elio/stdlib"
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

// TestCullAllBackends cross-validates the flagship frustum-culling kernel — the
// hard case the emitters must agree on: a uniform block, read + read_write
// storage buffers, an atomic array, arrayLength, a bounded loop with break, a
// matrix-column index, and swizzles. (CPU execution of this kernel, with a real
// frustum, lives in run.TestRunCull.)
func TestCullAllBackends(t *testing.T) {
	mod := ir.CullKernel()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "cull.wgsl", wsrc, func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "cull.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{
		"kernel void main(",
		"struct InstanceRecord",
		"device atomic_uint* drawArgs",
		"atomic_fetch_add_explicit",
	} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}
}

// TestReduceEmits validates the workgroup tree-reduction — Elio's first
// non-embarrassingly-parallel kernel — across the GPU backends. It is the proof
// for workgroup-shared memory and barriers: naga enforces that workgroupBarrier
// sits in uniform control flow, and glslang checks the shared/barrier GLSL. The
// CPU fallback intentionally cannot run it (run.Run rejects shared memory), so
// this kernel is GPU-only.
func TestReduceEmits(t *testing.T) {
	mod := ir.WorkgroupReduce()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	if !strings.Contains(wsrc, "var<workgroup> scratch") || !strings.Contains(wsrc, "workgroupBarrier()") {
		t.Errorf("wgsl missing shared/barrier:\n%s", wsrc)
	}
	validate(t, "naga", "reduce.wgsl", wsrc, func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	if !strings.Contains(gsrc, "shared float scratch[64];") || !strings.Contains(gsrc, "barrier();") {
		t.Errorf("glsl missing shared/barrier:\n%s", gsrc)
	}
	validate(t, "glslangValidator", "reduce.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"threadgroup float scratch[64];", "threadgroup_barrier(mem_flags::mem_threadgroup)"} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}
}

// TestScanEmits validates the stdlib workgroup prefix-sum across the GPU
// backends — a cooperative primitive with shared memory and barriers inside a
// log-step loop. (Its execution correctness is proven in stdlib by the lockstep
// interpreter; this guards the emitted source.)
func TestScanEmits(t *testing.T) {
	mod := stdlib.Scan()

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	if !strings.Contains(wsrc, "var<workgroup> temp") || !strings.Contains(wsrc, "workgroupBarrier()") {
		t.Errorf("wgsl missing shared/barrier:\n%s", wsrc)
	}
	validate(t, "naga", "scan.wgsl", wsrc, func(f string) []string { return []string{f} })

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "scan.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	for _, want := range []string{"threadgroup uint temp[64];", "threadgroup_barrier(mem_flags::mem_threadgroup)"} {
		if !strings.Contains(msrc, want) {
			t.Errorf("metal missing %q\n%s", want, msrc)
		}
	}
}

// TestCompactEmits validates the stdlib stream-compaction (scan + scatter) on
// the GPU backends. Its execution correctness is proven in stdlib.
func TestCompactEmits(t *testing.T) {
	mod := stdlib.Compact()
	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "compact.wgsl", wsrc, func(f string) []string { return []string{f} })
	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "compact.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })
	if _, err := metal.Emit(mod); err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
}

// TestSortEmits validates the stdlib bitonic sort. It is the strongest emitter
// check here: barriers nested inside two loops, which naga's uniformity analysis
// must accept. Execution correctness is proven in stdlib.
func TestSortEmits(t *testing.T) {
	mod := stdlib.Sort()
	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "sort.wgsl", wsrc, func(f string) []string { return []string{f} })
	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "sort.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })
	if _, err := metal.Emit(mod); err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
}

// TestParticleUpdateEmits validates the particle-simulation kernel — a uniform
// struct, a read_write storage array of structs, struct-field reads and writes
// (particles[i].px = …), and branching — across the GPU backends. Execution
// (integration + respawn) is proven in stdlib.
func TestParticleUpdateEmits(t *testing.T) {
	mod := stdlib.ParticleUpdate()
	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "particles.wgsl", wsrc, func(f string) []string { return []string{f} })
	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "particles.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })
	if _, err := metal.Emit(mod); err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
}

// TestHiZEmits validates the Hi-Z depth-pyramid downsample across the GPU
// backends — a uniform dims struct, two storage depth buffers, integer index
// math, and a 4-way max. Execution is proven in stdlib.
func TestHiZEmits(t *testing.T) {
	mod := stdlib.HiZDownsample()
	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	validate(t, "naga", "hiz.wgsl", wsrc, func(f string) []string { return []string{f} })
	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	validate(t, "glslangValidator", "hiz.comp", gsrc, func(f string) []string { return []string{"-V", f, "-S", "comp"} })
	if _, err := metal.Emit(mod); err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
}

// TestSkinEmits validates the linear-blend skinning kernel across the GPU
// backends — a storage array of mat4 bones, matrix-column indexing
// (bones[i][0]), vector arithmetic, and a struct of mixed f32/u32 fields.
// Execution (the 50/50 bone blend) is proven in stdlib.
func TestSkinEmits(t *testing.T) {
	mod := stdlib.Skin()
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
	if _, err := metal.Emit(mod); err != nil {
		t.Fatalf("metal.Emit: %v", err)
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
	cmd := exec.Command(path, args(f)...)
	cmd.Dir = dir // contain any default output (e.g. glslang's comp.spv) to the tempdir
	out, err := cmd.CombinedOutput()
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
