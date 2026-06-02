package stdlib

import "m31labs.dev/elio/ir"

// HiZDownsample returns the Hi-Z (hierarchical depth) pyramid build step: each
// destination texel is the max of the 2×2 source texels under it, halving the
// depth buffer's resolution. Run once per mip level, it produces the
// conservative depth pyramid a GPU occlusion-cull tests bounding boxes against —
// the M2 "Hi-Z occlusion" primitive.
//
// max (not min) makes the pyramid conservative for a reversed-Z / "farthest
// wins" convention: a box is occluded only if it is behind every covered texel.
// It is embarrassingly parallel — one lane per destination texel, integer index
// math, no shared memory.
func HiZDownsample() *ir.Module {
	i := ir.Name{Name: "i"}
	dim := func(field string) ir.Expr { return ir.Member{E: ir.Name{Name: "dims"}, Field: field} }
	bin := func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	u := func(s string) ir.Expr { return ir.Lit{Text: s} }
	// src[row * srcWidth + col]
	srcAt := func(col, row ir.Expr) ir.Expr {
		return ir.Index{E: ir.Name{Name: "src"}, Idx: bin("+", bin("*", row, dim("srcWidth")), col)}
	}
	mx := func(a, b ir.Expr) ir.Expr { return ir.Call{Func: "max", Args: []ir.Expr{a, b}} }
	sx, sy := ir.Name{Name: "sx"}, ir.Name{Name: "sy"}

	return &ir.Module{
		Structs: []ir.Struct{
			{Name: "HiZDims", Fields: []ir.Field{{Name: "srcWidth", Type: ir.U32}, {Name: "dstWidth", Type: ir.U32}}},
		},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "dims", Type: ir.Named{Name: "HiZDims"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Array{Elem: ir.F32}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.ReadWrite, Name: "dst", Type: ir.Array{Elem: ir.F32}},
		},
		Kernels: []ir.Kernel{{
			Name:          "hiz",
			WorkgroupSize: [3]int{tileWidth, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
				// destination (col,row) → top-left source texel (2x, 2y)
				ir.Let{Name: "sx", Value: bin("*", bin("%", i, dim("dstWidth")), u("2u"))},
				ir.Let{Name: "sy", Value: bin("*", bin("/", i, dim("dstWidth")), u("2u"))},
				ir.Let{Name: "d0", Value: srcAt(sx, sy)},
				ir.Let{Name: "d1", Value: srcAt(bin("+", sx, u("1u")), sy)},
				ir.Let{Name: "d2", Value: srcAt(sx, bin("+", sy, u("1u")))},
				ir.Let{Name: "d3", Value: srcAt(bin("+", sx, u("1u")), bin("+", sy, u("1u")))},
				ir.Assign{
					Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: i},
					Value:  mx(mx(ir.Name{Name: "d0"}, ir.Name{Name: "d1"}), mx(ir.Name{Name: "d2"}, ir.Name{Name: "d3"})),
				},
			},
		}},
	}
}
