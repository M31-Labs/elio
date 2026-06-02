package ir

import "strconv"

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
//	binding 4  outPos   array<vec4<f32>>  read_write  skinned position; .w =
//	                                                   Σ(weights·col3.w·p.w)
//	                                                   (==1 for normalized affine
//	                                                   rigs), not a passthrough
//
// Joint indices are TRUSTED: out-of-range indices are undefined behavior (GPU) /
// out-of-bounds (CPU). The caller must supply valid bone indices; only the vertex
// dimension is bounds-checked (via arrayLength).
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

// SkinDQS returns the IR for a dual-quaternion skinning (DQS) compute kernel —
// the rotation-preserving alternative to linear-blend skinning, free of the
// candy-wrapper collapse LBS suffers at twisting joints. For each vertex it
// blends up to four bones' unit dual quaternions (with antipodality handling),
// normalizes the result, and applies the rigid transform plus a blended uniform
// scale to the rest position.
//
// Buffer contract (all group 0, address space storage):
//
//	binding 0  restPos    array<vec4<f32>>  read        rest position, xyz pos, w == 1
//	binding 1  joints     array<vec4<u32>>  read        four bone indices per vertex
//	binding 2  weights    array<vec4<f32>>  read        four influence weights (>= 0)
//	binding 3  realQ      array<vec4<f32>>  read        per-bone real quaternion (xyzw),
//	                                                     indexed DIRECTLY by bone index
//	binding 4  dualQ      array<vec4<f32>>  read        per-bone dual quaternion (xyzw)
//	binding 5  boneScale  array<vec4<f32>>  read        per-bone uniform scale in .x..z
//	binding 6  dst        array<vec4<f32>>  read_write  skinned position; .w == 1
//
// The palette is pre-built (real/dual quats per bone) by the host, so the kernel
// stays inside Elio's op set: it needs only vec×scalar, vec+vec, dot, and the
// sqrt builtin (added for the normalize step), plus componentwise member-assign
// to write dst[i].{x,y,z,w}. Bone arrays are indexed by the bone index directly
// (NOT ×4 as LBS's column-major matrix palette is). Influence 0's weight is
// assumed non-zero, so it seeds the accumulators and defines the antipodality
// reference quaternion `ref`.
func SkinDQS() *Module {
	// refQ = realQ[j.x]: the antipodality reference (influence 0's real quat).
	// Named refQ (not ref) because `ref` is a reserved keyword in WGSL.
	refQ := Index{Name{"realQ"}, Member{Name{"j"}, "x"}}

	// Seed the accumulators from influence 0 (component x) — no flip needed since
	// dot(ref, ref) >= 0. Initializing from the first influence avoids needing a
	// zero-vector constructor.
	body := []Stmt{
		Let{"i", Member{Name{"gid"}, "x"}},
		If{
			Cond: Binary{">=", Name{"i"}, Call{"arrayLength", []Expr{AddrOf{Name{"restPos"}}}}},
			Then: []Stmt{Return{}},
		},
		Let{"p", Index{Name{"restPos"}, Name{"i"}}},
		Let{"j", Index{Name{"joints"}, Name{"i"}}},
		Let{"w", Index{Name{"weights"}, Name{"i"}}},
		Let{"refQ", refQ},
		Var{Name: "accReal", Type: Vec{4, F32}, Init: Binary{"*", Index{Name{"realQ"}, Member{Name{"j"}, "x"}}, Member{Name{"w"}, "x"}}},
		Var{Name: "accDual", Type: Vec{4, F32}, Init: Binary{"*", Index{Name{"dualQ"}, Member{Name{"j"}, "x"}}, Member{Name{"w"}, "x"}}},
		Var{Name: "scaleAcc", Type: Vec{4, F32}, Init: Binary{"*", Index{Name{"boneScale"}, Member{Name{"j"}, "x"}}, Member{Name{"w"}, "x"}}},
		Var{Name: "wsum", Type: F32, Init: Member{Name{"w"}, "x"}},
	}

	// Accumulate influences 1..3 (components y, z, w) with antipodality flip.
	for n, comp := range []string{"y", "z", "w"} {
		s := strconv.Itoa(n)
		bc := "bc" + s
		rc := "rc" + s
		dc := "dc" + s
		sc := "sc" + s
		wq := "wq" + s
		body = append(body,
			Let{bc, Member{Name{"j"}, comp}},
			Let{rc, Index{Name{"realQ"}, Name{bc}}},
			Let{dc, Index{Name{"dualQ"}, Name{bc}}},
			Let{sc, Index{Name{"boneScale"}, Name{bc}}},
			Var{Name: wq, Type: F32, Init: Member{Name{"w"}, comp}},
			If{
				Cond: Binary{"<", Call{"dot", []Expr{Name{rc}, Name{"refQ"}}}, Lit{"0.0"}},
				Then: []Stmt{Assign{Target: Name{wq}, Value: Unary{"-", Member{Name{"w"}, comp}}}},
			},
			Assign{Target: Name{"accReal"}, Value: Binary{"+", Name{"accReal"}, Binary{"*", Name{rc}, Name{wq}}}},
			Assign{Target: Name{"accDual"}, Value: Binary{"+", Name{"accDual"}, Binary{"*", Name{dc}, Name{wq}}}},
			Assign{Target: Name{"scaleAcc"}, Value: Binary{"+", Name{"scaleAcc"}, Binary{"*", Name{sc}, Member{Name{"w"}, comp}}}},
			Assign{Target: Name{"wsum"}, Value: Binary{"+", Name{"wsum"}, Member{Name{"w"}, comp}}},
		)
	}

	// Normalize the blended dual quaternion by the real part's magnitude.
	body = append(body,
		Let{"inv", Binary{"/", Lit{"1.0"}, Call{"sqrt", []Expr{Call{"dot", []Expr{Name{"accReal"}, Name{"accReal"}}}}}}},
		Let{"r", Binary{"*", Name{"accReal"}, Name{"inv"}}},
		Let{"d", Binary{"*", Name{"accDual"}, Name{"inv"}}},
		// Blended uniform scale, applied componentwise to the rest position.
		Let{"invw", Binary{"/", Lit{"1.0"}, Name{"wsum"}}},
		Let{"sb", Binary{"*", Name{"scaleAcc"}, Name{"invw"}}},
		Let{"base", Binary{"*", Name{"p"}, Name{"sb"}}},
	)

	// Rotate base.xyz by the unit quaternion r (u = r.xyz):
	//   uv  = cross(u, base.xyz)
	//   uuv = cross(u, uv)
	//   rotated = base.xyz + 2*r.w*uv + 2*uuv
	rx := Member{Name{"r"}, "x"}
	ry := Member{Name{"r"}, "y"}
	rz := Member{Name{"r"}, "z"}
	rw := Member{Name{"r"}, "w"}
	bx := Member{Name{"base"}, "x"}
	by := Member{Name{"base"}, "y"}
	bz := Member{Name{"base"}, "z"}

	// cross(a,b) = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
	// uv = cross(r.xyz, base.xyz)
	body = append(body,
		Let{"uvx", Binary{"-", Binary{"*", ry, bz}, Binary{"*", rz, by}}},
		Let{"uvy", Binary{"-", Binary{"*", rz, bx}, Binary{"*", rx, bz}}},
		Let{"uvz", Binary{"-", Binary{"*", rx, by}, Binary{"*", ry, bx}}},
		// uuv = cross(r.xyz, uv)
		Let{"uuvx", Binary{"-", Binary{"*", ry, Name{"uvz"}}, Binary{"*", rz, Name{"uvy"}}}},
		Let{"uuvy", Binary{"-", Binary{"*", rz, Name{"uvx"}}, Binary{"*", rx, Name{"uvz"}}}},
		Let{"uuvz", Binary{"-", Binary{"*", rx, Name{"uvy"}}, Binary{"*", ry, Name{"uvx"}}}},
		// rotated = base + 2*r.w*uv + 2*uuv
		Let{"rotx", Binary{"+", bx, Binary{"+", Binary{"*", Binary{"*", Lit{"2.0"}, rw}, Name{"uvx"}}, Binary{"*", Lit{"2.0"}, Name{"uuvx"}}}}},
		Let{"roty", Binary{"+", by, Binary{"+", Binary{"*", Binary{"*", Lit{"2.0"}, rw}, Name{"uvy"}}, Binary{"*", Lit{"2.0"}, Name{"uuvy"}}}}},
		Let{"rotz", Binary{"+", bz, Binary{"+", Binary{"*", Binary{"*", Lit{"2.0"}, rw}, Name{"uvz"}}, Binary{"*", Lit{"2.0"}, Name{"uuvz"}}}}},
	)

	// Translation t = xyz of 2 * qMul(d, conj(r)):
	//   tx = 2*( -d.w*r.x + d.x*r.w - d.y*r.z + d.z*r.y )
	//   ty = 2*( -d.w*r.y + d.x*r.z + d.y*r.w - d.z*r.x )
	//   tz = 2*( -d.w*r.z - d.x*r.y + d.y*r.x + d.z*r.w )
	dx := Member{Name{"d"}, "x"}
	dy := Member{Name{"d"}, "y"}
	dz := Member{Name{"d"}, "z"}
	dw := Member{Name{"d"}, "w"}
	body = append(body,
		Let{"tx", Binary{"*", Lit{"2.0"},
			Binary{"+", Binary{"-", Binary{"+", Unary{"-", Binary{"*", dw, rx}}, Binary{"*", dx, rw}}, Binary{"*", dy, rz}}, Binary{"*", dz, ry}}}},
		Let{"ty", Binary{"*", Lit{"2.0"},
			Binary{"-", Binary{"+", Binary{"+", Unary{"-", Binary{"*", dw, ry}}, Binary{"*", dx, rz}}, Binary{"*", dy, rw}}, Binary{"*", dz, rx}}}},
		Let{"tz", Binary{"*", Lit{"2.0"},
			Binary{"+", Binary{"+", Binary{"-", Unary{"-", Binary{"*", dw, rz}}, Binary{"*", dx, ry}}, Binary{"*", dy, rx}}, Binary{"*", dz, rw}}}},
	)

	// Write the skinned position: dst[i].{x,y,z} = rotated + t, dst[i].w = 1.
	dsti := Index{Name{"dst"}, Name{"i"}}
	body = append(body,
		Assign{Target: Member{dsti, "x"}, Value: Binary{"+", Name{"rotx"}, Name{"tx"}}},
		Assign{Target: Member{dsti, "y"}, Value: Binary{"+", Name{"roty"}, Name{"ty"}}},
		Assign{Target: Member{dsti, "z"}, Value: Binary{"+", Name{"rotz"}, Name{"tz"}}},
		Assign{Target: Member{dsti, "w"}, Value: Lit{"1.0"}},
	)

	return &Module{
		Bindings: []Binding{
			{Group: 0, Binding: 0, Space: Storage, Access: Read, Name: "restPos", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 1, Space: Storage, Access: Read, Name: "joints", Type: Array{Elem: Vec{4, U32}}},
			{Group: 0, Binding: 2, Space: Storage, Access: Read, Name: "weights", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 3, Space: Storage, Access: Read, Name: "realQ", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 4, Space: Storage, Access: Read, Name: "dualQ", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 5, Space: Storage, Access: Read, Name: "boneScale", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 6, Space: Storage, Access: ReadWrite, Name: "dst", Type: Array{Elem: Vec{4, F32}}},
		},
		Kernels: []Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: Vec{3, U32}}},
			Body:          body,
		}},
	}
}

// SqrtKernel is a minimal kernel exercising the sqrt builtin (the elio language
// addition that dual-quaternion skinning needs for its normalize step). For
// each input vec4 p it writes out[i] = p * sqrt(dot(p,p)) = p * |p|. With
// p = {3,4,0,0}, |p| = 5, so out = {15,20,0,0}. Validates that sqrt emits on
// every backend (WGSL/GLSL/Metal name it `sqrt`) and runs on the CPU oracle.
func SqrtKernel() *Module {
	return &Module{
		Bindings: []Binding{
			{Group: 0, Binding: 0, Space: Storage, Access: Read, Name: "a", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 1, Space: Storage, Access: ReadWrite, Name: "dst", Type: Array{Elem: Vec{4, F32}}},
		},
		Kernels: []Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: Vec{3, U32}}},
			Body: []Stmt{
				Let{"i", Member{Name{"gid"}, "x"}},
				If{Cond: Binary{">=", Name{"i"}, Call{"arrayLength", []Expr{AddrOf{Name{"a"}}}}}, Then: []Stmt{Return{}}},
				Let{"p", Index{Name{"a"}, Name{"i"}}},
				Let{"s", Call{"sqrt", []Expr{Call{"dot", []Expr{Name{"p"}, Name{"p"}}}}}},
				Assign{Target: Index{Name{"dst"}, Name{"i"}}, Value: Binary{"*", Name{"p"}, Name{"s"}}},
			},
		}},
	}
}
