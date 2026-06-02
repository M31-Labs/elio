package stdlib

import "m31labs.dev/elio/ir"

// Skin returns linear-blend (4-influence) vertex skinning: each vertex is
// transformed by up to four bone matrices and blended by its weights, the
// per-frame deform a skinned mesh needs before it is drawn. It is the last M2
// consumer ("skinning") and the first matrix-heavy Elio kernel.
//
// The bone transform is expressed by column expansion —
// m[0]*px + m[1]*py + m[2]*pz + m[3] — rather than a mat*vec product, so it uses
// only matrix-column indexing and vector ops Elio already has (no mat*vec
// builtin, no vec4 constructor). Bone indices are u32 so they index the palette
// directly; weights are f32. Output is three floats per vertex (xyz).
func Skin() *ir.Module {
	i := ir.Name{Name: "i"}
	v := func(field string) ir.Expr { return ir.Member{E: ir.Name{Name: "v"}, Field: field} }
	bin := func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	u := func(s string) ir.Expr { return ir.Lit{Text: s} }
	add := func(parts ...ir.Expr) ir.Expr {
		e := parts[0]
		for _, p := range parts[1:] {
			e = bin("+", e, p)
		}
		return e
	}
	// bones[v.<boneField>][col]
	col := func(boneField, c string) ir.Expr {
		return ir.Index{E: ir.Index{E: ir.Name{Name: "bones"}, Idx: v(boneField)}, Idx: u(c)}
	}
	// weighted transform of v's position by the bone in boneField, scaled by wField:
	//   v.wField * (col0*px + col1*py + col2*pz + col3)
	influence := func(boneField, wField string) ir.Expr {
		xf := add(
			bin("*", col(boneField, "0u"), v("px")),
			bin("*", col(boneField, "1u"), v("py")),
			bin("*", col(boneField, "2u"), v("pz")),
			col(boneField, "3u"),
		)
		return bin("*", xf, v(wField))
	}
	skinnedComp := func(c string) ir.Expr { return ir.Member{E: ir.Name{Name: "skinned"}, Field: c} }
	outAt := func(o string) ir.Expr {
		return ir.Index{E: ir.Name{Name: "out"}, Idx: bin("+", bin("*", i, u("3u")), u(o))}
	}

	f32 := func(n string) ir.Field { return ir.Field{Name: n, Type: ir.F32} }
	u32 := func(n string) ir.Field { return ir.Field{Name: n, Type: ir.U32} }

	return &ir.Module{
		Structs: []ir.Struct{
			{Name: "SkinVertex", Fields: []ir.Field{
				f32("px"), f32("py"), f32("pz"),
				f32("w0"), f32("w1"), f32("w2"), f32("w3"),
				u32("b0"), u32("b1"), u32("b2"), u32("b3"),
			}},
		},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "bones", Type: ir.Array{Elem: ir.Mat{Cols: 4, Rows: 4, Elem: ir.F32}}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "verts", Type: ir.Array{Elem: ir.Named{Name: "SkinVertex"}}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.ReadWrite, Name: "out", Type: ir.Array{Elem: ir.F32}},
		},
		Kernels: []ir.Kernel{{
			Name:          "skin",
			WorkgroupSize: [3]int{tileWidth, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
				ir.Let{Name: "v", Value: ir.Index{E: ir.Name{Name: "verts"}, Idx: i}},
				ir.Let{Name: "skinned", Value: add(
					influence("b0", "w0"), influence("b1", "w1"),
					influence("b2", "w2"), influence("b3", "w3"),
				)},
				ir.Assign{Target: outAt("0u"), Value: skinnedComp("x")},
				ir.Assign{Target: outAt("1u"), Value: skinnedComp("y")},
				ir.Assign{Target: outAt("2u"), Value: skinnedComp("z")},
			},
		}},
	}
}
