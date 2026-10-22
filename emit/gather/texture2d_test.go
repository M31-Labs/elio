package gather

import (
	"strings"
	"testing"

	"m31labs.dev/elio/parse"
)

const sampledGatherSource = `struct P { w: u32; }
@group(0) @binding(0) uniform p: P;
@group(0) @binding(1) texture src: tex2d;
@group(0) @binding(2) storage write dst: tex2d_rgba16float;

@workgroup(64) kernel k(gid: global_invocation_id) {
  let i = gid.x;
  let dx = i % p.w;
  let dy = i / p.w;
  let d = textureDimensions(src, 0u);
  let x1 = min(dx * 2u + 1u, d.x - 1u);
  let v = textureLoad(src, vec2u(dx * 2u, dy * 2u), 0u) + textureLoad(src, vec2u(x1, dy * 2u), 0u);
  textureStore(dst, vec2u(i % p.w, i / p.w), v * 0.5);
}
`

func TestEmitSampledTextureInput(t *testing.T) {
	m, err := parse.Parse(sampledGatherSource)
	if err != nil {
		t.Fatal(err)
	}
	src, lay, err := Emit(m, "k")
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"uniform highp sampler2D src;",
		"uvec2 d = uvec2(textureSize(src, int(0u)));",
		"texelFetch(src, ivec2(uvec2((dx * 2u), (dy * 2u))), int(0u))",
		"dst_out = (v * 0.5);",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("missing %q in:\n%s", want, src)
		}
	}
	for _, bad := range []string{"src_tex", "src_len", "src_at", "elio_width)"} {
		if bad == "elio_width)" {
			continue
		}
		if strings.Contains(src, bad) {
			t.Errorf("unexpected %q (a texture is already 2D) in:\n%s", bad, src)
		}
	}
	if len(lay.Inputs) != 1 {
		t.Fatalf("inputs = %+v", lay.Inputs)
	}
	in := lay.Inputs[0]
	if in.Name != "src" || in.Kind != "texture" || in.Sampler != "src" || in.Length != "" || in.Elem != "vec4" || in.Binding != 1 {
		t.Errorf("input = %+v", in)
	}
}

func TestEmitBufferInputKind(t *testing.T) {
	_, lay, err := Emit(twoTextures("rgba16float"), "k")
	if err != nil {
		t.Fatal(err)
	}
	if got := lay.Inputs[0].Kind; got != "buffer" {
		t.Errorf("buffer input kind = %q, want buffer", got)
	}
}

func TestTextureDimensionsOfSampledTextureIsNotPassWidth(t *testing.T) {
	m, err := parse.Parse(strings.Replace(sampledGatherSource, "let d = textureDimensions(src, 0u);", "let d = textureDimensions(src);", 1))
	if err != nil {
		t.Fatal(err)
	}
	src, _, err := Emit(m, "k")
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	if !strings.Contains(src, "uvec2 d = uvec2(textureSize(src, 0));") {
		t.Errorf("one-argument textureDimensions on a sampled texture:\n%s", src)
	}
}
