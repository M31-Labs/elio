package gather

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

// twoTextures reads src and writes two RGBA16F textures, one with the width
// from a uniform field and one with textureDimensions(t).x.
func twoTextures(format string) *ir.Module {
	n := func(s string) ir.Expr { return ir.Name{Name: s} }
	gx := ir.Member{E: n("gid"), Field: "x"}
	coord := func(w ir.Expr) ir.Expr {
		return ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Binary{Op: "%", L: n("i"), R: w}, ir.Binary{Op: "/", L: n("i"), R: w}}}
	}
	store := func(tex string, w ir.Expr, v ir.Expr) ir.Stmt {
		return ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{n(tex), coord(w), v}}}
	}
	return &ir.Module{
		Structs: []ir.Struct{{Name: "P", Fields: []ir.Field{{Name: "w", Type: ir.U32}, {Name: "k", Type: ir.F32}}}},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "p", Type: ir.Named{Name: "P"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.Write, Name: "disp", Type: ir.StorageTexture{Format: format}},
			{Group: 0, Binding: 3, Space: ir.Storage, Access: ir.Write, Name: "slope", Type: ir.StorageTexture{Format: format}},
		},
		Kernels: []ir.Kernel{{
			Name:          "k",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: gx},
				ir.Let{Name: "v", Value: ir.Index{E: n("src"), Idx: n("i")}},
				store("disp", ir.Member{E: n("p"), Field: "w"}, ir.Binary{Op: "*", L: n("v"), R: ir.Member{E: n("p"), Field: "k"}}),
				store("slope", ir.Member{E: ir.Call{Func: "textureDimensions", Args: []ir.Expr{n("slope")}}, Field: "x"}, n("v")),
			},
		}},
	}
}

func TestEmitTextureOutputs(t *testing.T) {
	src, lay, err := Emit(twoTextures("rgba16float"), "k")
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"layout(location = 0) out vec4 disp_out;",
		"layout(location = 1) out vec4 slope_out;",
		"disp_out = (v * p.k);",
		"slope_out = v;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("missing %q in:\n%s", want, src)
		}
	}
	for _, bad := range []string{"textureStore", "textureDimensions", "slope_tex", "disp_tex"} {
		if strings.Contains(src, bad) {
			t.Errorf("unexpected %q in:\n%s", bad, src)
		}
	}
	if len(lay.Inputs) != 1 || len(lay.Outputs) != 2 {
		t.Fatalf("layout = %+v", lay)
	}
	for i, name := range []string{"disp", "slope"} {
		o := lay.Outputs[i]
		if o.Name != name || o.Format != "rgba16float" || o.Elem != "vec4" || o.Location != i || o.Variable != name+"_out" {
			t.Errorf("output %d = %+v", i, o)
		}
	}
}

func TestEmitBufferOutputHasNoFormat(t *testing.T) {
	_, lay, err := Emit(ir.ScaleBias(), "main")
	if err != nil {
		t.Fatal(err)
	}
	if lay.Outputs[0].Format != "" {
		t.Errorf("buffer output format = %q, want empty", lay.Outputs[0].Format)
	}
}

func TestEmitRejectsThreeFloatTextures(t *testing.T) {
	m := twoTextures("rgba32float")
	m.Bindings = append(m.Bindings, ir.Binding{Group: 0, Binding: 4, Space: ir.Storage, Access: ir.Write, Name: "third", Type: ir.StorageTexture{Format: "rgba32float"}})
	body := m.Kernels[0].Body
	m.Kernels[0].Body = append(body, ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{
		ir.Name{Name: "third"},
		ir.Call{Func: "vec2u", Args: []ir.Expr{
			ir.Binary{Op: "%", L: ir.Name{Name: "i"}, R: ir.Member{E: ir.Name{Name: "p"}, Field: "w"}},
			ir.Binary{Op: "/", L: ir.Name{Name: "i"}, R: ir.Member{E: ir.Name{Name: "p"}, Field: "w"}}}},
		ir.Name{Name: "v"}}}})
	if _, _, err := Emit(m, "k"); err == nil || !strings.Contains(err.Error(), "32 bytes per pixel") {
		t.Fatalf("want a 32 bytes per pixel error, got %v", err)
	}
}
