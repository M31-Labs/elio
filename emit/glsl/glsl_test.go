package glsl

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

// TestEmitCullGLSL pins the cull kernel's GLSL shape: inlined uniform block,
// std430 storage buffers, reserved-word escaping (input→input_), the
// type-inferred local declarations GLSL needs, runtime length via .length(),
// and the &-free atomic.
func TestEmitCullGLSL(t *testing.T) {
	src, err := Emit(ir.CullKernel())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"#version 450",
		"struct InstanceRecord {",
		"mat4 model;",
		"layout(std140, binding = 0) uniform cull_ubo {",
		"vec4 planes[6];",
		"} cull;",
		"layout(std430, binding = 1) readonly buffer input_ssbo { InstanceRecord input_[]; };",
		"layout(std430, binding = 3) buffer drawArgs_ssbo { uint drawArgs[]; };",
		"layout(local_size_x = 64) in;",
		"uint i = gl_GlobalInvocationID.x;", // inferred type + builtin rename
		"if ((i >= input_.length())) {",     // arrayLength → .length(), escaped
		"InstanceRecord record = input_[i];",
		"vec3 center = record.model[3].xyz;", // inferred vec3
		"bool inside = true;",                // inferred bool
		"for (int p = 0; (p < 6); p = (p + 1)) {",
		"vec4 plane = cull.planes[p];",            // inferred vec4
		"uint slot = atomicAdd(drawArgs[1], 1u);", // &-free atomic
		"output_[slot] = record;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted GLSL missing %q\n--- got ---\n%s", want, src)
		}
	}
	if strings.Contains(src, "__") {
		t.Errorf("emitted GLSL contains reserved consecutive underscores:\n%s", src)
	}
}

// TestGLSLValidatesToSPIRV compiles the emitted GLSL with glslangValidator,
// proving it is real, SPIR-V-ready Vulkan compute source. Skips if glslang is
// not installed.
func TestGLSLValidatesToSPIRV(t *testing.T) {
	bin, err := exec.LookPath("glslangValidator")
	if err != nil {
		t.Skip("glslangValidator not installed")
	}
	src, err := Emit(ir.CullKernel())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	dir := t.TempDir()
	comp := filepath.Join(dir, "cull.comp")
	if err := os.WriteFile(comp, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(bin, "-V", comp, "-S", "comp", "-o", filepath.Join(dir, "cull.spv")).CombinedOutput()
	if err != nil {
		t.Fatalf("glslangValidator failed: %v\n%s\n--- source ---\n%s", err, out, src)
	}
	if strings.Contains(string(out), "ERROR") || strings.Contains(string(out), "WARNING") {
		t.Errorf("glslang reported issues:\n%s", out)
	}
}

func TestEmitStorageTexture(t *testing.T) {
	for format, glf := range map[string]string{"rgba16float": "rgba16f", "rgba32float": "rgba32f", "r32float": "r32f"} {
		src, err := Emit(ir.TextureFill(format))
		if err != nil {
			t.Fatalf("Emit: %v", err)
		}
		for _, want := range []string{
			"layout(" + glf + ", binding = 2) uniform writeonly image2D disp;",
			"imageStore(disp, ivec2(uvec2(x, y)), vec4(float(x), float(y), 0.5, 1.0));",
		} {
			if !strings.Contains(src, want) {
				t.Errorf("%s: GLSL missing %q\n--- got ---\n%s", format, want, src)
			}
		}
	}
	m := ir.TextureFill("rgba16float")
	m.Kernels[0].Body = append(m.Kernels[0].Body, ir.Let{Name: "d", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{ir.Name{Name: "disp"}}}})
	src, _ := Emit(m)
	if !strings.Contains(src, "uvec2 d = uvec2(imageSize(disp));") {
		t.Errorf("textureDimensions not lowered:\n%s", src)
	}
}

func TestEmitSampledTexture(t *testing.T) {
	src, err := Emit(ir.TextureCopy())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"layout(binding = 0) uniform sampler2D src;",
		"vec4 v = texelFetch(src, ivec2(c), int(0u));",
		"uvec2 d = uvec2(textureSize(src, int(1u)));",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("GLSL missing %q\n--- got ---\n%s", want, src)
		}
	}
	m := ir.TextureCopy()
	m.Kernels[0].Body = append(m.Kernels[0].Body, ir.Let{Name: "d0", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{ir.Name{Name: "src"}}}})
	src, _ = Emit(m)
	if !strings.Contains(src, "uvec2 d0 = uvec2(textureSize(src, 0));") {
		t.Errorf("one-argument textureDimensions on a sampled texture not lowered:\n%s", src)
	}
}
