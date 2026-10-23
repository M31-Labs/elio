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

func TestEmitStorageTexture(t *testing.T) {
	src, err := Emit(ir.TextureFill("rgba16float"))
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"texture2d<float, access::write> disp [[texture(0)]]",
		"disp.write(float4(float(x), float(y), 0.5, 1.0), uint2(uint2(x, y)));",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("MSL missing %q\n--- got ---\n%s", want, src)
		}
	}
	// Textures count separately from buffers: a buffer before the texture keeps
	// buffer(0), and the texture is still texture(0).
	m := ir.TextureFill("r32float")
	m.Bindings = append([]ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "buf", Type: ir.Array{Elem: ir.F32}}}, m.Bindings...)
	src, _ = Emit(m)
	for _, want := range []string{"device float* buf [[buffer(0)]]", "disp [[texture(0)]]"} {
		if !strings.Contains(src, want) {
			t.Errorf("MSL missing %q\n%s", want, src)
		}
	}
	m.Kernels[0].Body = append(m.Kernels[0].Body, ir.Let{Name: "d", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{ir.Name{Name: "disp"}}}})
	src, _ = Emit(m)
	if !strings.Contains(src, "uint2(disp.get_width(), disp.get_height())") {
		t.Errorf("textureDimensions not lowered:\n%s", src)
	}
}

func TestEmitSampledTexture(t *testing.T) {
	src, err := Emit(ir.TextureCopy())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"texture2d<float, access::read> src [[texture(0)]]",
		"texture2d<float, access::write> dst [[texture(1)]]",
		"src.read(uint2(c), 0u)",
		"uint2(src.get_width(1u), src.get_height(1u))",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("MSL missing %q\n--- got ---\n%s", want, src)
		}
	}
}
