package gather

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

const wantFragIndex = "(uint(gl_FragCoord.y) * elio_width + uint(gl_FragCoord.x))"

func TestEmitScaleBias(t *testing.T) {
	src, lay, err := Emit(ir.ScaleBias(), "main")
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	for _, want := range []string{
		"#version 300 es",
		"precision highp float;",
		"precision highp int;",
		"precision highp sampler2D;",
		"uniform uint elio_width;",
		"uniform highp sampler2D src_tex;",
		"uniform uint src_len;",
		"uniform uint src_w;",
		"layout(std140) uniform Params_ubo {",
		"} params;",
		"vec4 src_at(uint i) {",
		"texelFetch(src_tex, ivec2(int(i % src_w), int(i / src_w)), 0)",
		"layout(location = 0) out vec4 dst_out;",
		"dst_out = vec4(0.0);",
		"uint i = " + wantFragIndex + ";",
		"if ((i >= src_len)) {",
		"dst_out = ((src_at(i) * params.scale) + params.bias);",
		"return;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("missing %q in:\n%s", want, src)
		}
	}
	if len(lay.Inputs) != 1 || len(lay.Outputs) != 1 || len(lay.Uniforms) != 1 {
		t.Fatalf("layout = %+v", lay)
	}
	if got := lay.Inputs[0]; got.Name != "src" || got.Elem != "vec4" || got.Sampler != "src_tex" || got.Length != "src_len" || got.Width != "src_w" {
		t.Errorf("input = %+v", got)
	}
	if got := lay.Outputs[0]; got.Name != "dst" || got.Elem != "vec4" || got.Location != 0 || got.Variable != "dst_out" {
		t.Errorf("output = %+v", got)
	}
	if got := lay.Uniforms[0]; got.Name != "params" || got.Block != "Params_ubo" || got.Struct != "Params" {
		t.Errorf("uniform = %+v", got)
	}
	if lay.Kernel != "main" {
		t.Errorf("kernel = %q", lay.Kernel)
	}
}

func TestEmitPadsNarrowOutputs(t *testing.T) {
	m := &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "a", Type: ir.Array{Elem: ir.Vec{N: 2, Elem: ir.F32}}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "b", Type: ir.Array{Elem: ir.F32}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.ReadWrite, Name: "c", Type: ir.Array{Elem: ir.Vec{N: 2, Elem: ir.F32}}},
		},
		Kernels: []ir.Kernel{{
			Name:          "k",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "v", Value: ir.Index{E: ir.Name{Name: "a"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}}},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "b"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}}, Value: ir.Member{E: ir.Name{Name: "v"}, Field: "x"}},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "c"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}}, Value: ir.Name{Name: "v"}},
			},
		}},
	}
	src, lay, err := Emit(m, "k")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"vec2 v = a_at(" + wantFragIndex + ");",
		"b_out = vec4(v.x, 0.0, 0.0, 0.0);",
		"c_out = vec4(v, 0.0, 0.0);",
		"layout(location = 0) out vec4 b_out;",
		"layout(location = 1) out vec4 c_out;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("missing %q in:\n%s", want, src)
		}
	}
	if lay.Outputs[1].Location != 1 || lay.Outputs[0].Elem != "f32" || lay.Outputs[1].Elem != "vec2" {
		t.Errorf("outputs = %+v", lay.Outputs)
	}
}
