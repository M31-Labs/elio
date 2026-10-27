package sema

import (
	"testing"

	"m31labs.dev/elio/ir"
)

func wTex(slot int, name, format string) ir.Binding {
	return ir.Binding{Group: 0, Binding: slot, Space: ir.Storage, Access: ir.Write, Name: name, Type: ir.StorageTexture{Format: format}}
}

func nm(s string) ir.Expr { return ir.Name{Name: s} }

// texCoord is vec2u(i % W, i / W).
func texCoord(i, w ir.Expr) ir.Expr {
	return ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Binary{Op: "%", L: i, R: w}, ir.Binary{Op: "/", L: i, R: w}}}
}

func storeTex(tex string, coord ir.Expr) ir.Stmt {
	return ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{nm(tex), coord,
		ir.Call{Func: "vec4", Args: []ir.Expr{ir.Lit{Text: "1.0"}, ir.Lit{Text: "0.0"}, ir.Lit{Text: "0.0"}, ir.Lit{Text: "1.0"}}}}}}
}

var texParams = ir.Binding{Group: 0, Binding: 9, Space: ir.Uniform, Name: "p", Type: ir.Named{Name: "P"}}

func gtexModule(body []ir.Stmt, texs ...ir.Binding) *ir.Module {
	m := gatherKernel(append([]ir.Binding{texParams}, texs...), body...)
	m.Structs = []ir.Struct{{Name: "P", Fields: []ir.Field{{Name: "w", Type: ir.U32}}}}
	return m
}

func TestCheckGatherTwoHalfTextures(t *testing.T) {
	w := ir.Member{E: nm("p"), Field: "w"}
	m := gtexModule([]ir.Stmt{
		ir.Let{Name: "i", Value: gidX()},
		storeTex("a", texCoord(nm("i"), w)),
		storeTex("b", texCoord(gidX(), ir.Member{E: ir.Call{Func: "textureDimensions", Args: []ir.Expr{nm("b")}}, Field: "x"})),
	}, wTex(0, "a", "rgba16float"), wTex(1, "b", "rgba16float"))
	if errs := CheckGather(m, "k"); len(errs) != 0 {
		t.Fatalf("want no diagnostics, got %v", errs)
	}
}

func TestCheckGatherThreeFloatTexturesExceedBytes(t *testing.T) {
	w := ir.Member{E: nm("p"), Field: "w"}
	m := gtexModule([]ir.Stmt{
		ir.Let{Name: "i", Value: gidX()},
		storeTex("a", texCoord(nm("i"), w)),
		storeTex("b", texCoord(nm("i"), w)),
		storeTex("c", texCoord(nm("i"), w)),
	}, wTex(0, "a", "rgba32float"), wTex(1, "b", "rgba32float"), wTex(2, "c", "rgba32float"))
	errs := CheckGather(m, "k")
	if !hasDiag(errs, "48", "32 bytes per pixel") {
		t.Fatalf("want a 32 bytes per pixel diagnostic, got %v", errs)
	}
}

func TestCheckGatherBufferOutputCountsAsRGBA32F(t *testing.T) {
	w := ir.Member{E: nm("p"), Field: "w"}
	m := gtexModule([]ir.Stmt{
		ir.Let{Name: "i", Value: gidX()},
		storeTex("a", texCoord(nm("i"), w)),
		storeTex("b", texCoord(nm("i"), w)),
		ir.Assign{Target: ir.Index{E: nm("buf"), Idx: nm("i")}, Value: ir.Lit{Text: "1.0"}},
	}, wTex(0, "a", "rgba32float"), wTex(1, "b", "rgba16float"), rwBuf(2, "buf", ir.F32))
	if errs := CheckGather(m, "k"); !hasDiag(errs, "40", "32 bytes per pixel") {
		t.Fatalf("want 40 bytes diagnostic, got %v", errs)
	}
}

func TestCheckGatherTextureCoordinateRules(t *testing.T) {
	w := ir.Member{E: nm("p"), Field: "w"}
	for name, body := range map[string][]ir.Stmt{
		"swapped": {storeTex("a", ir.Call{Func: "vec2u", Args: []ir.Expr{
			ir.Binary{Op: "/", L: gidX(), R: w}, ir.Binary{Op: "%", L: gidX(), R: w}}})},
		"not gid":       {ir.Let{Name: "j", Value: ir.Lit{Text: "3u"}}, storeTex("a", texCoord(nm("j"), w))},
		"literal width": {storeTex("a", texCoord(gidX(), ir.Lit{Text: "4u"}))},
		"bare dimensions": {ir.Let{Name: "d", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{nm("a")}}},
			storeTex("a", texCoord(gidX(), ir.Member{E: nm("d"), Field: "x"}))},
	} {
		m := gtexModule(body, wTex(0, "a", "rgba16float"))
		if errs := CheckGather(m, "k"); !hasDiag(errs, "rule 3", `"a"`) {
			t.Errorf("%s: want a rule 3 diagnostic naming the texture, got %v", name, errs)
		}
	}
}

func TestCheckGatherSampledTextureInput(t *testing.T) {
	w := ir.Member{E: nm("p"), Field: "w"}
	src := ir.Binding{Group: 0, Binding: 5, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Texture2D{}}
	load := ir.Call{Func: "textureLoad", Args: []ir.Expr{nm("src"), ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Lit{Text: "0u"}, ir.Lit{Text: "0u"}}}, ir.Lit{Text: "0u"}}}
	dims := ir.Call{Func: "textureDimensions", Args: []ir.Expr{nm("src"), ir.Lit{Text: "0u"}}}
	m := gtexModule([]ir.Stmt{
		ir.Let{Name: "i", Value: gidX()},
		ir.Let{Name: "d", Value: dims},
		ir.Let{Name: "v", Value: load},
		ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{nm("a"), texCoord(nm("i"), w), nm("v")}}},
	}, src, wTex(0, "a", "rgba16float"))
	if errs := CheckGather(m, "k"); len(errs) != 0 {
		t.Fatalf("sampled texture input rejected: %v", errs)
	}
	if errs := Check(m); len(errs) != 0 {
		t.Fatalf("Check: %v", Errors(errs))
	}
}
