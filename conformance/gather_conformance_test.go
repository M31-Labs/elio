package conformance

import (
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/gather"
	"m31labs.dev/elio/ir"
)

func validateES(t *testing.T, src string) {
	t.Helper()
	prismvalidate.Shader(t, "glslangValidator", src, ".frag", func(f string) []string { return []string{"-S", "frag", f} })
}

// gatherMix reads two buffers (vec4 and vec2), writes two (f32 and vec4), and
// uses a uniform block with u32, f32 and vector fields, a loop, casts, select
// and arrayLength.
func gatherMix() *ir.Module {
	gx := ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}
	n := func(s string) ir.Expr { return ir.Name{Name: s} }
	return &ir.Module{
		Structs: []ir.Struct{{Name: "MixParams", Fields: []ir.Field{
			{Name: "taps", Type: ir.U32},
			{Name: "gain", Type: ir.F32},
			{Name: "tint", Type: ir.Vec{N: 3, Elem: ir.F32}},
			{Name: "offset", Type: ir.Vec{N: 2, Elem: ir.F32}},
		}}},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "mp", Type: ir.Named{Name: "MixParams"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "a", Type: ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.Read, Name: "b", Type: ir.Array{Elem: ir.Vec{N: 2, Elem: ir.F32}}},
			{Group: 0, Binding: 3, Space: ir.Storage, Access: ir.ReadWrite, Name: "sum", Type: ir.Array{Elem: ir.F32}},
			{Group: 0, Binding: 4, Space: ir.Storage, Access: ir.ReadWrite, Name: "mixed", Type: ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}}},
		},
		Kernels: []ir.Kernel{{
			Name:          "mix",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: gx},
				ir.If{
					Cond: ir.Binary{Op: ">=", L: n("i"), R: ir.Call{Func: "arrayLength", Args: []ir.Expr{ir.AddrOf{E: n("a")}}}},
					Then: []ir.Stmt{ir.Return{}},
				},
				ir.Var{Name: "acc", Init: ir.Lit{Text: "0.0"}},
				ir.For{
					Init: ir.Var{Name: "t", Type: ir.U32, Init: ir.Lit{Text: "0u"}},
					Cond: ir.Binary{Op: "<", L: n("t"), R: ir.Member{E: n("mp"), Field: "taps"}},
					Post: ir.Assign{Target: n("t"), Value: ir.Binary{Op: "+", L: n("t"), R: ir.Lit{Text: "1u"}}},
					Body: []ir.Stmt{
						ir.Assign{Target: n("acc"), Op: "+", Value: ir.Member{E: ir.Index{E: n("b"), Idx: ir.Binary{Op: "%", L: ir.Binary{Op: "+", L: n("i"), R: n("t")}, R: ir.Call{Func: "arrayLength", Args: []ir.Expr{ir.AddrOf{E: n("b")}}}}}, Field: "y"}},
					},
				},
				ir.Let{Name: "va", Value: ir.Index{E: n("a"), Idx: n("i")}},
				ir.Let{Name: "w", Value: ir.Call{Func: "select", Args: []ir.Expr{ir.Lit{Text: "0.5"}, ir.Lit{Text: "1.0"}, ir.Binary{Op: ">", L: ir.Member{E: n("va"), Field: "w"}, R: ir.Lit{Text: "0.0"}}}}},
				ir.Assign{Target: ir.Index{E: n("sum"), Idx: n("i")}, Value: ir.Binary{Op: "*", L: n("acc"), R: ir.Member{E: n("mp"), Field: "gain"}}},
				ir.Assign{
					Target: ir.Index{E: n("mixed"), Idx: n("i")},
					Value: ir.Call{Func: "vec4", Args: []ir.Expr{
						ir.Binary{Op: "*", L: ir.Member{E: n("va"), Field: "xyz"}, R: ir.Member{E: n("mp"), Field: "tint"}},
						n("w"),
					}},
				},
			},
		}},
	}
}

// gatherTwoTextures reads one buffer and writes two RGBA16F storage textures.
func gatherTwoTextures() *ir.Module {
	n := func(s string) ir.Expr { return ir.Name{Name: s} }
	coord := func(w ir.Expr) ir.Expr {
		return ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Binary{Op: "%", L: n("i"), R: w}, ir.Binary{Op: "/", L: n("i"), R: w}}}
	}
	store := func(tex string, w, v ir.Expr) ir.Stmt {
		return ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{n(tex), coord(w), v}}}
	}
	return &ir.Module{
		Structs: []ir.Struct{{Name: "TexParams", Fields: []ir.Field{{Name: "w", Type: ir.U32}, {Name: "k", Type: ir.F32}}}},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "tp", Type: ir.Named{Name: "TexParams"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.Write, Name: "disp", Type: ir.StorageTexture{Format: "rgba16float"}},
			{Group: 0, Binding: 3, Space: ir.Storage, Access: ir.Write, Name: "slope", Type: ir.StorageTexture{Format: "rgba16float"}},
		},
		Kernels: []ir.Kernel{{
			Name:          "two",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: ir.Member{E: n("gid"), Field: "x"}},
				ir.If{
					Cond: ir.Binary{Op: ">=", L: n("i"), R: ir.Call{Func: "arrayLength", Args: []ir.Expr{ir.AddrOf{E: n("src")}}}},
					Then: []ir.Stmt{ir.Return{}},
				},
				ir.Let{Name: "v", Value: ir.Index{E: n("src"), Idx: n("i")}},
				store("disp", ir.Member{E: n("tp"), Field: "w"}, ir.Binary{Op: "*", L: n("v"), R: ir.Member{E: n("tp"), Field: "k"}}),
				store("slope", ir.Member{E: ir.Call{Func: "textureDimensions", Args: []ir.Expr{n("slope")}}, Field: "x"}, n("v")),
			},
		}},
	}
}

func TestGatherESValidates(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mod    *ir.Module
		kernel string
		in     int
		out    int
	}{
		{"scalebias", ir.ScaleBias(), "main", 1, 1},
		{"mix", gatherMix(), "mix", 2, 2},
		{"two textures", gatherTwoTextures(), "two", 1, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src, lay, err := gather.Emit(tc.mod, tc.kernel)
			if err != nil {
				t.Fatalf("gather.Emit: %v", err)
			}
			if len(lay.Inputs) != tc.in || len(lay.Outputs) != tc.out {
				t.Errorf("layout inputs=%d outputs=%d, want %d and %d", len(lay.Inputs), len(lay.Outputs), tc.in, tc.out)
			}
			validateES(t, src)
		})
	}
}

// TestGatherESValidatorIsReal shows the check can fail: a missing semicolon and
// a missing float precision (an ES-only error) must both be reported.
func TestGatherESValidatorIsReal(t *testing.T) {
	src, _, err := gather.Emit(ir.ScaleBias(), "main")
	if err != nil {
		t.Fatal(err)
	}
	argv := func(f string) []string { return []string{"-S", "frag", f} }
	for name, bad := range map[string]string{
		"syntax":       strings.Replace(src, "return;", "return", 1),
		"no precision": strings.Replace(src, "precision highp float;\n", "", 1),
	} {
		if bad == src {
			t.Fatalf("%s: mutation did not change the source", name)
		}
		r, err := prismvalidate.Run("glslangValidator", bad, ".frag", argv)
		if r.Skipped {
			t.Skip("glslangValidator not on PATH")
		}
		if err == nil && !strings.Contains(r.Output, "ERROR") {
			t.Errorf("%s: validator accepted a broken shader:\n%s", name, r.Output)
		}
	}
}
