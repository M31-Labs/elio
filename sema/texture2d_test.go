package sema

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

func sampledModule(body ...ir.Stmt) *ir.Module {
	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Texture2D{}},
		},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{16, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body:     body,
		}},
	}
}

func vec2u(a, b string) ir.Expr {
	return ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Lit{Text: a}, ir.Lit{Text: b}}}
}

func TestSampledTexture(t *testing.T) {
	load := ir.Let{Name: "v", Value: ir.Call{Func: "textureLoad", Args: []ir.Expr{ir.Name{Name: "src"}, vec2u("1u", "2u"), ir.Lit{Text: "0u"}}}}
	dims := ir.Let{Name: "d", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{ir.Name{Name: "src"}, ir.Lit{Text: "1u"}}}}
	if errs := Check(ir.TextureCopy()); len(errs) != 0 {
		t.Fatalf("TextureCopy rejected:\n%v", Errors(errs))
	}
	if errs := Check(sampledModule(load, dims)); len(errs) != 0 {
		t.Fatalf("valid sampled texture module rejected:\n%v", Errors(errs))
	}

	cases := []struct {
		name string
		mod  *ir.Module
		want string
	}{
		{"assign", sampledModule(ir.Assign{Target: ir.Name{Name: "src"}, Value: ir.Lit{Text: "1"}}), `cannot assign to "src"`},
		{"address", sampledModule(ir.Let{Name: "p", Value: ir.AddrOf{E: ir.Name{Name: "src"}}}), `cannot take the address of "src"`},
		{"level not u32 literal", sampledModule(ir.Let{Name: "v", Value: ir.Call{Func: "textureLoad", Args: []ir.Expr{
			ir.Name{Name: "src"}, vec2u("1u", "2u"), ir.Lit{Text: "0"}}}}), "level must be u32"},
		{"level float", sampledModule(ir.Let{Name: "d", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{
			ir.Name{Name: "src"}, ir.Lit{Text: "0.0"}}}}), "level must be u32"},
		{"level typed local", sampledModule(
			ir.Var{Name: "l", Type: ir.I32, Init: ir.Lit{Text: "0"}},
			ir.Let{Name: "v", Value: ir.Call{Func: "textureLoad", Args: []ir.Expr{ir.Name{Name: "src"}, vec2u("1u", "2u"), ir.Name{Name: "l"}}}}),
			"level must be u32"},
		{"load arity", sampledModule(ir.Let{Name: "v", Value: ir.Call{Func: "textureLoad", Args: []ir.Expr{
			ir.Name{Name: "src"}, vec2u("1u", "2u")}}}), "textureLoad expects 3 argument(s), got 2"},
		{"load from storage texture", texModule("rgba16float", ir.Let{Name: "v", Value: ir.Call{Func: "textureLoad", Args: []ir.Expr{
			ir.Name{Name: "disp"}, vec2u("1u", "2u"), ir.Lit{Text: "0u"}}}}), `textureLoad needs a sampled texture`},
		{"store to sampled", sampledModule(ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{
			ir.Name{Name: "src"}, vec2u("1u", "2u"),
			ir.Call{Func: "vec4", Args: []ir.Expr{ir.Lit{Text: "1.0"}, ir.Lit{Text: "0.0"}, ir.Lit{Text: "0.0"}, ir.Lit{Text: "1.0"}}}}}}),
			"textureStore needs a storage texture"},
		{"write access", &ir.Module{Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Write, Name: "src", Type: ir.Texture2D{}},
		}}, "sampled texture needs read access"},
		{"read_write access", &ir.Module{Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "src", Type: ir.Texture2D{}},
		}}, "sampled texture needs read access"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Errors(Check(tc.mod))
			if got == nil || !strings.Contains(got.Error(), tc.want) {
				t.Fatalf("want diagnostic %q, got %v", tc.want, got)
			}
		})
	}
}
