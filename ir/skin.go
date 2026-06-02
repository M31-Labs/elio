package ir

// SkinLBS returns the IR for a linear-blend skinning (LBS) compute kernel — the
// GPU side of Kiln's skeletal animation. For each vertex it transforms the rest
// position by up to four weighted bone matrices and writes the blended result.
//
// Buffer contract (all group 0, address space storage):
//
//	binding 0  restPos  array<vec4<f32>>  read        rest position, xyz pos, w == 1
//	binding 1  joints   array<vec4<u32>>  read        four bone indices per vertex
//	binding 2  weights  array<vec4<f32>>  read        four influence weights per vertex
//	binding 3  palette  array<vec4<f32>>  read        bone matrices, COLUMN-MAJOR:
//	                                                   bone j's four columns live at
//	                                                   palette[j*4+0 .. j*4+3]
//	binding 4  outPos   array<vec4<f32>>  read_write  skinned position
//
// Storing the palette column-major is what lets the kernel stay inside Elio's
// current op set: a matrix·point product M·p is just
//
//	col0*p.x + col1*p.y + col2*p.z + col3*p.w   (p.w == 1)
//
// which uses only vec×scalar broadcast and vec+vec — both already supported by
// the interpreter (run.vecBinop) and every emitter. No dot/sqrt/abs/bitwise op
// is needed. Each of the four influences is transformed by its bone's columns
// and scaled by its weight; the four are accumulated into outPos.
func SkinLBS() *Module {
	body := []Stmt{
		Let{"i", Member{Name{"gid"}, "x"}},
		If{
			Cond: Binary{">=", Name{"i"}, Call{"arrayLength", []Expr{AddrOf{Name{"restPos"}}}}},
			Then: []Stmt{Return{}},
		},
		Let{"p", Index{Name{"restPos"}, Name{"i"}}},
		Let{"j", Index{Name{"joints"}, Name{"i"}}},
		Let{"w", Index{Name{"weights"}, Name{"i"}}},
	}

	// First influence seeds the accumulator; the remaining three add into it.
	bx, ex := weightedInfluence("x")
	body = append(body, bx, Var{Name: "acc", Type: Vec{4, F32}, Init: ex})
	for _, comp := range []string{"y", "z", "w"} {
		b, e := weightedInfluence(comp)
		body = append(body, b, Assign{Target: Name{"acc"}, Value: Binary{"+", Name{"acc"}, e}})
	}
	body = append(body, Assign{Target: Index{Name{"outPos"}, Name{"i"}}, Value: Name{"acc"}})

	return &Module{
		Bindings: []Binding{
			{Group: 0, Binding: 0, Space: Storage, Access: Read, Name: "restPos", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 1, Space: Storage, Access: Read, Name: "joints", Type: Array{Elem: Vec{4, U32}}},
			{Group: 0, Binding: 2, Space: Storage, Access: Read, Name: "weights", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 3, Space: Storage, Access: Read, Name: "palette", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 4, Space: Storage, Access: ReadWrite, Name: "outPos", Type: Array{Elem: Vec{4, F32}}},
		},
		Kernels: []Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: Vec{3, U32}}},
			Body:          body,
		}},
	}
}

// col is bone column c of the matrix whose first column is palette[base].
func col(base Expr, c string) Expr {
	return Index{Name{"palette"}, Binary{"+", base, Lit{c}}}
}

// transform applies the column-major matrix at base to point p:
// col0*p.x + col1*p.y + col2*p.z + col3*p.w.
func transform(base Expr) Expr {
	t := Binary{"*", col(base, "0u"), Member{Name{"p"}, "x"}}
	t2 := Binary{"+", t, Binary{"*", col(base, "1u"), Member{Name{"p"}, "y"}}}
	t3 := Binary{"+", t2, Binary{"*", col(base, "2u"), Member{Name{"p"}, "z"}}}
	return Binary{"+", t3, Binary{"*", col(base, "3u"), Member{Name{"p"}, "w"}}}
}

// weightedInfluence returns (a let binding the base palette column index for the
// bone in joint component comp, and the transformed point scaled by weight comp).
func weightedInfluence(comp string) (Stmt, Expr) {
	baseName := "b_" + comp
	return Let{baseName, Binary{"*", Member{Name{"j"}, comp}, Lit{"4u"}}},
		Binary{"*", transform(Name{baseName}), Member{Name{"w"}, comp}}
}
