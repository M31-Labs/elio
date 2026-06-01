package wgsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

// TestEmitCull proves the Elio compiler reproduces the frustum-cull kernel that
// examples/cull hand-writes: build the IR, emit WGSL, compare to the golden.
// Regenerate the golden with UPDATE_GOLDEN=1 go test ./emit/wgsl/.
func TestEmitCull(t *testing.T) {
	src, err := Emit(ir.CullKernel())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}

	golden := filepath.Join("testdata", "cull.wgsl")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(golden, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run UPDATE_GOLDEN=1 to create): %v", err)
	}
	if src != string(want) {
		t.Errorf("emitted WGSL != golden\n--- got ---\n%s\n--- want ---\n%s", src, string(want))
	}
}

// TestEmitCullStructure pins the key constructs independent of formatting, so a
// golden churn can't silently drop a load-bearing piece of the kernel.
func TestEmitCullStructure(t *testing.T) {
	src, err := Emit(ir.CullKernel())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"struct CullUniforms {",
		"struct InstanceRecord {",
		"@group(0) @binding(0) var<uniform> cull : CullUniforms;",
		"@group(0) @binding(1) var<storage, read> input : array<InstanceRecord>;",
		"@group(0) @binding(3) var<storage, read_write> drawArgs : array<atomic<u32>, 4>;",
		"@compute @workgroup_size(64)",
		"fn main(@builtin(global_invocation_id) gid : vec3<u32>) {",
		"arrayLength(&input)",
		"for (var p : i32 = 0; (p < 6); p = (p + 1)) {",
		"atomicAdd(&drawArgs[1], 1u)",
		"output[slot] = record;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted WGSL missing %q\n%s", want, src)
		}
	}
}
