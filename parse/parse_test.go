package parse

import (
	"os"
	"path/filepath"
	"testing"

	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
)

// TestParseCullMatchesGolden is the front-end proof: parse cull.elio into IR,
// emit WGSL, and assert it is byte-identical to the WGSL backend's golden (which
// is itself naga-validated and generated from the hand-built ir.CullKernel()).
// Source → IR → WGSL therefore lands exactly where the hand-built IR does.
func TestParseCullMatchesGolden(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "cull.elio"))
	if err != nil {
		t.Fatal(err)
	}
	mod, err := Parse(string(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	want, err := os.ReadFile(filepath.Join("..", "emit", "wgsl", "testdata", "cull.wgsl"))
	if err != nil {
		t.Fatalf("read wgsl golden: %v", err)
	}
	if got != string(want) {
		t.Errorf("parsed-then-emitted WGSL != hand-built golden\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

// TestParseEqualsHandBuilt pins that the parsed module is structurally identical
// to the hand-built one (same emitted WGSL from both paths).
func TestParseEqualsHandBuilt(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "cull.elio"))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(string(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	fromParse, _ := wgsl.Emit(parsed)
	fromHand, _ := wgsl.Emit(ir.CullKernel())
	if fromParse != fromHand {
		t.Errorf("parsed IR emits differently than hand-built IR\n--- parsed ---\n%s\n--- hand ---\n%s", fromParse, fromHand)
	}
}

func TestParseStorageTextureBinding(t *testing.T) {
	for _, format := range []string{"rgba16float", "rgba32float", "r32float"} {
		src := "@group(0) @binding(2) storage write disp: tex2d_" + format + ";\n" +
			"@workgroup(16) kernel main(gid: global_invocation_id) {\n" +
			"  textureStore(disp, vec2u(gid.x, 0u), vec4(1.0, 0.0, 0.0, 1.0));\n}\n"
		mod, err := Parse(src)
		if err != nil {
			t.Fatalf("%s: Parse: %v", format, err)
		}
		if len(mod.Bindings) != 1 {
			t.Fatalf("%s: got %d bindings", format, len(mod.Bindings))
		}
		if body := mod.Kernels[0].Body; len(body) != 1 {
			t.Fatalf("%s: want 1 statement, got %d", format, len(body))
		} else if do, ok := body[0].(ir.Do); !ok {
			t.Fatalf("%s: statement is %T, want ir.Do", format, body[0])
		} else if call, ok := do.Expr.(ir.Call); !ok || call.Func != "textureStore" || len(call.Args) != 3 {
			t.Fatalf("%s: bad call %+v", format, do.Expr)
		}
		want := ir.Binding{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.Write, Name: "disp", Type: ir.StorageTexture{Format: format}}
		if mod.Bindings[0] != want {
			t.Errorf("%s: binding = %+v, want %+v", format, mod.Bindings[0], want)
		}
	}
}
