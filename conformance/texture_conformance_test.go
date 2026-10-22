package conformance

import (
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/parse"
	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

const textureFillSource = `@group(0) @binding(2) storage write disp: tex2d_rgba16float;

@workgroup(16) kernel main(gid: global_invocation_id) {
  let x = gid.x % 4u;
  let y = gid.x / 4u;
  textureStore(disp, vec2u(x, y), vec4(f32(x), f32(y), 0.5, 1.0));
}
`

// TestStorageTextureAllBackends runs the 4x4 texture-fill kernel through every
// backend, from hand-built IR and from .elio source: naga validates WGSL,
// glslangValidator compiles GLSL, Metal is checked structurally, and the
// interpreter executes it.
func TestStorageTextureAllBackends(t *testing.T) {
	parsed, err := parse.Parse(textureFillSource)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	for _, tc := range []struct {
		name   string
		mod    *ir.Module
		format string
		glFmt  string
	}{
		{"rgba16float", ir.TextureFill("rgba16float"), "rgba16float", "rgba16f"},
		{"rgba32float", ir.TextureFill("rgba32float"), "rgba32float", "rgba32f"},
		{"r32float", ir.TextureFill("r32float"), "r32float", "r32f"},
		{"parsed", parsed, "rgba16float", "rgba16f"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if errs := sema.Check(tc.mod); len(errs) != 0 {
				t.Fatalf("sema: %v", sema.Errors(errs))
			}
			wsrc, err := wgsl.Emit(tc.mod)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(wsrc, "texture_storage_2d<"+tc.format+", write>") {
				t.Errorf("WGSL lacks the texture declaration:\n%s", wsrc)
			}
			gsrc, err := glsl.Emit(tc.mod)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(gsrc, "layout("+tc.glFmt+", binding = 2) uniform writeonly image2D disp;") {
				t.Errorf("GLSL lacks the image declaration:\n%s", gsrc)
			}
			msrc, err := metal.Emit(tc.mod)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"texture2d<float, access::write> disp [[texture(0)]]", "disp.write("} {
				if !strings.Contains(msrc, want) {
					t.Errorf("Metal lacks %q:\n%s", want, msrc)
				}
			}
			t.Run("naga", func(t *testing.T) {
				prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(f string) []string { return []string{f} })
			})
			t.Run("glslang", func(t *testing.T) {
				prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(f string) []string { return []string{"-V", f, "-S", "comp"} })
			})

			tex := &run.Texture{Width: 4, Height: 4, Format: tc.format, Data: make([]float64, 64)}
			if err := run.Run(tc.mod, "main", 16, &run.Memory{Vars: map[string]any{"disp": tex}}); err != nil {
				t.Fatalf("run: %v", err)
			}
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					i := (y*4 + x) * 4
					want := []float64{float64(x), float64(y), 0.5, 1}
					for c := 0; c < 4; c++ {
						if tex.Data[i+c] != want[c] {
							t.Fatalf("texel(%d,%d) = %v, want %v", x, y, tex.Data[i:i+4], want)
						}
					}
				}
			}
		})
	}
}

const sampledTextureSource = `@group(0) @binding(0) texture src: tex2d;
@group(0) @binding(1) storage write dst: tex2d_rgba32float;

@workgroup(16) kernel main(gid: global_invocation_id) {
  let x = gid.x % 4u;
  let y = gid.x / 4u;
  let c = vec2u(x, y);
  let d = textureDimensions(src, 1u);
  let v = textureLoad(src, c, 0u);
  textureStore(dst, c, v + vec4(f32(d.x), 0.0, 0.0, 0.0));
}
`

// TestSampledTextureAllBackends runs a kernel that reads a mip chain through
// textureLoad and textureDimensions(t, level), from IR and from source.
func TestSampledTextureAllBackends(t *testing.T) {
	parsed, err := parse.Parse(sampledTextureSource)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	for name, mod := range map[string]*ir.Module{"ir": ir.TextureCopy(), "parsed": parsed} {
		t.Run(name, func(t *testing.T) {
			if errs := sema.Check(mod); len(errs) != 0 {
				t.Fatalf("sema: %v", sema.Errors(errs))
			}
			wsrc, err := wgsl.Emit(mod)
			if err != nil {
				t.Fatal(err)
			}
			gsrc, err := glsl.Emit(mod)
			if err != nil {
				t.Fatal(err)
			}
			msrc, err := metal.Emit(mod)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"var src : texture_2d<f32>;", "textureLoad(src, vec2<i32>(c), i32(0u))"} {
				if !strings.Contains(wsrc, want) {
					t.Errorf("WGSL lacks %q:\n%s", want, wsrc)
				}
			}
			for _, want := range []string{"uniform sampler2D src;", "texelFetch(src, ivec2(c), int(0u))"} {
				if !strings.Contains(gsrc, want) {
					t.Errorf("GLSL lacks %q:\n%s", want, gsrc)
				}
			}
			for _, want := range []string{"texture2d<float, access::read> src [[texture(0)]]", "src.read(uint2(c), 0u)"} {
				if !strings.Contains(msrc, want) {
					t.Errorf("Metal lacks %q:\n%s", want, msrc)
				}
			}
			t.Run("naga", func(t *testing.T) {
				prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(f string) []string { return []string{f} })
			})
			t.Run("glslang", func(t *testing.T) {
				prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(f string) []string { return []string{"-V", f, "-S", "comp"} })
			})

			base := &run.Texture{Width: 4, Height: 4, Format: "rgba32float", Data: make([]float64, 64)}
			for i := 0; i < 16; i++ {
				copy(base.Data[i*4:], []float64{float64(i), float64(i) * 2, 0, 1})
			}
			base.Levels = []*run.Texture{{Width: 2, Height: 2, Format: "rgba32float", Data: make([]float64, 16)}}
			dst := &run.Texture{Width: 4, Height: 4, Format: "rgba32float", Data: make([]float64, 64)}
			if err := run.Run(mod, "main", 16, &run.Memory{Vars: map[string]any{"src": base, "dst": dst}}); err != nil {
				t.Fatalf("run: %v", err)
			}
			for i := 0; i < 16; i++ {
				want := []float64{float64(i) + 2, float64(i) * 2, 0, 1}
				for c := 0; c < 4; c++ {
					if dst.Data[i*4+c] != want[c] {
						t.Fatalf("texel %d = %v, want %v", i, dst.Data[i*4:i*4+4], want)
					}
				}
			}
		})
	}
}
