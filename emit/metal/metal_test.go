package metal

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

// TestEmitCullMetal asserts the cull kernel lowers to the expected MSL shape:
// kernel-argument bindings with [[buffer(i)]], pointer storage buffers, the
// synthesized arrayLength → length-argument rewrite, the atomic intrinsic, and
// the matrix-column swizzle. (No MSL validator on Linux CI, so this pins the
// structure; the WGSL backend is naga-validated and the CPU backend executes.)
func TestEmitCullMetal(t *testing.T) {
	src, err := Emit(ir.CullKernel())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"#include <metal_stdlib>",
		"using namespace metal;",
		"struct CullUniforms {",
		"float4 planes[6];", // MSL array-field syntax (size after name)
		"float4x4 model;",
		"kernel void main(",
		"constant CullUniforms& cull [[buffer(0)]]",
		"const device InstanceRecord* input [[buffer(1)]]",
		"device InstanceRecord* output [[buffer(2)]]",
		"device atomic_uint* drawArgs [[buffer(3)]]",
		"constant uint& inputLength [[buffer(4)]]", // synthesized for arrayLength
		"uint3 gid [[thread_position_in_grid]]",
		"if ((i >= inputLength)) {", // arrayLength(&input) rewritten
		"auto record = input[i];",
		"record.model[3].xyz",
		"for (int p = 0; (p < 6); p = (p + 1)) {",
		"dot(plane.xyz, center)",
		"atomic_fetch_add_explicit(&drawArgs[1], 1u, memory_order_relaxed)",
		"output[slot] = record;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted MSL missing %q\n--- got ---\n%s", want, src)
		}
	}
}
