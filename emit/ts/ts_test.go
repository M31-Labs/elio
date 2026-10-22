package ts

import (
	"os"
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/parse"
)

const miniSrc = `struct P { n: u32; k: f32; }
@group(0) @binding(0) uniform p: P;
@group(0) @binding(1) storage read src: []vec4;
@group(0) @binding(2) storage write dst: tex2d_rgba16float;
@workgroup(64) kernel mini(gid: global_invocation_id) {
  let i = gid.x;
  if i >= p.n { return; }
  textureStore(dst, vec2u(i % p.n, i / p.n), src[i] * p.k);
}
`

func miniSet(t *testing.T, gather bool) KernelSet {
	t.Helper()
	m, err := parse.Parse(miniSrc)
	if err != nil {
		t.Fatal(err)
	}
	return KernelSet{Name: "mini", Kernels: []Kernel{{Name: "mini_kernel", Entry: "mini", Module: m, Gather: gather}}}
}

func TestEmitExports(t *testing.T) {
	out, err := Emit(miniSet(t, true), Meta{Version: "v-test"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"// Elio v-test", "// sha256: ",
		"export const miniKernelWGSL = `", "export const miniKernelEntry = \"mini\";",
		"export const miniKernelWorkgroup = [64, 1, 1];",
		"export const miniKernelBindings", "storage-texture", "rgba16float",
		"export const miniKernelUniforms", "\"offset\": 4",
		"export const miniKernelGLSL = `", "export const miniKernelGatherLayout",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q", want)
		}
	}
	again, _ := Emit(miniSet(t, true), Meta{Version: "v-test"})
	if again != out {
		t.Error("output is not deterministic")
	}
	noGather, err := Emit(miniSet(t, false), Meta{Version: "v-test"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(noGather, "KernelGLSL") {
		t.Error("non-gather kernel emitted GLSL")
	}
}

func TestEmitRejectsBadGather(t *testing.T) {
	m, _ := parse.Parse(`@group(0) @binding(0) storage read_write d: []f32;
@workgroup(64) kernel k(gid: global_invocation_id) { d[0u] = 1.0; }`)
	_, err := Emit(KernelSet{Name: "x", Kernels: []Kernel{{Name: "k", Entry: "k", Module: m, Gather: true}}}, Meta{})
	if err == nil {
		t.Fatal("want gather check error")
	}
	_, err = Emit(KernelSet{Name: "x", Kernels: []Kernel{{Name: "k", Entry: "nope", Module: &ir.Module{}}}}, Meta{})
	if err == nil {
		t.Fatal("want missing entry error")
	}
}

func TestEscape(t *testing.T) {
	in := "a`b${c}\\d"
	if got := UnescapeTemplate(escapeTemplate(in)); got != in {
		t.Errorf("round trip = %q", got)
	}
	if strings.Contains(escapeTemplate(in), "`b") && !strings.Contains(escapeTemplate(in), "\\`") {
		t.Error("backtick not escaped")
	}
}

func golden(t *testing.T, path, got string) {
	t.Helper()
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(want) != got {
		t.Errorf("%s differs from emitted output; rerun with UPDATE_GOLDEN=1", path)
	}
}

func TestMiniGolden(t *testing.T) {
	out, err := Emit(miniSet(t, true), Meta{Version: "v-test"})
	if err != nil {
		t.Fatal(err)
	}
	golden(t, "testdata/mini.ts", out)
}

func TestGlobalMini(t *testing.T) {
	out, err := Emit(miniSet(t, true), Meta{Version: "v-test", Global: "__test_elio"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"window.__test_elio = deepFreeze({", "miniKernel: {", "wgsl: `", "entry: \"mini\",",
		"workgroup: [64, 1, 1],", "bindings: [", "glsl: `", "gatherLayout: {",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q", want)
		}
	}
	if strings.Contains(out, "export ") || strings.Contains(out, ": ElioBinding[]") {
		t.Error("global form carries TypeScript syntax")
	}
	es, err := Emit(miniSet(t, true), Meta{Version: "v-test"})
	if err != nil {
		t.Fatal(err)
	}
	if hashLine(es) != hashLine(out) {
		t.Error("global and ES forms disagree on the sha256 of the sources")
	}
	golden(t, "testdata/mini.global.js", out)
	if _, err := Emit(miniSet(t, true), Meta{Global: "a.b"}); err == nil {
		t.Error("want error for a dotted global name")
	}
}

func hashLine(s string) string {
	for _, l := range strings.Split(s, "\n") {
		if strings.HasPrefix(l, "// sha256: ") {
			return l
		}
	}
	return ""
}
