package ir

// CullKernel returns the IR for the frustum-culling compute kernel — the exact
// kernel examples/cull hand-writes in WGSL. It exercises Elio's imperative
// surface end-to-end: structs, a uniform block + storage buffers, atomics, a
// bounded loop, if / break / return, a mutable local, buffer indexing, and a
// matrix-column swizzle. Emitting it is the M0→Move-3 proof that the Elio
// compiler reproduces the hand-written kernel.
func CullKernel() *Module {
	return &Module{
		Structs: []Struct{
			{Name: "CullUniforms", Fields: []Field{
				{"planes", Array{Elem: Vec{4, F32}, Len: 6}},
				{"vertexCount", U32},
				{"radius", F32},
				{"_pad0", Vec{2, F32}},
			}},
			{Name: "InstanceRecord", Fields: []Field{
				{"model", Mat{4, 4, F32}},
				{"pickData", Vec{4, U32}},
			}},
		},
		Bindings: []Binding{
			{Group: 0, Binding: 0, Space: Uniform, Name: "cull", Type: Named{"CullUniforms"}},
			{Group: 0, Binding: 1, Space: Storage, Access: Read, Name: "input", Type: Array{Elem: Named{"InstanceRecord"}}},
			{Group: 0, Binding: 2, Space: Storage, Access: ReadWrite, Name: "output", Type: Array{Elem: Named{"InstanceRecord"}}},
			{Group: 0, Binding: 3, Space: Storage, Access: ReadWrite, Name: "drawArgs", Type: Array{Elem: Atomic{U32}, Len: 4}},
		},
		Kernels: []Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: Vec{3, U32}}},
			Body: []Stmt{
				Let{"i", Member{Name{"gid"}, "x"}},
				If{
					Cond: Binary{">=", Name{"i"}, Call{"arrayLength", []Expr{AddrOf{Name{"input"}}}}},
					Then: []Stmt{Return{}},
				},
				Let{"record", Index{Name{"input"}, Name{"i"}}},
				Let{"center", Member{Index{Member{Name{"record"}, "model"}, Lit{"3"}}, "xyz"}},
				Var{Name: "inside", Init: Lit{"true"}},
				For{
					Init: Var{Name: "p", Type: I32, Init: Lit{"0"}},
					Cond: Binary{"<", Name{"p"}, Lit{"6"}},
					Post: Assign{Name{"p"}, Binary{"+", Name{"p"}, Lit{"1"}}},
					Body: []Stmt{
						Let{"plane", Index{Member{Name{"cull"}, "planes"}, Name{"p"}}},
						If{
							Cond: Binary{"<",
								Binary{"+",
									Call{"dot", []Expr{Member{Name{"plane"}, "xyz"}, Name{"center"}}},
									Member{Name{"plane"}, "w"}},
								Unary{"-", Member{Name{"cull"}, "radius"}}},
							Then: []Stmt{Assign{Name{"inside"}, Lit{"false"}}, Break{}},
						},
					},
				},
				If{
					Cond: Name{"inside"},
					Then: []Stmt{
						Let{"slot", Call{"atomicAdd", []Expr{AddrOf{Index{Name{"drawArgs"}, Lit{"1"}}}, Lit{"1u"}}}},
						Assign{Index{Name{"output"}, Name{"slot"}}, Name{"record"}},
					},
				},
			},
		}},
	}
}

// ScaleBias returns a second, structurally different kernel —
// dst[i] = src[i] * params.scale + params.bias over vec4 buffers — to prove the
// IR, emitters, and CPU interpreter generalize beyond cull. It exercises vector
// arithmetic (vec*vec, vec+vec) and a uniform struct of vectors, neither of
// which the cull kernel uses.
func ScaleBias() *Module {
	return &Module{
		Structs: []Struct{
			{Name: "Params", Fields: []Field{
				{"scale", Vec{4, F32}},
				{"bias", Vec{4, F32}},
			}},
		},
		Bindings: []Binding{
			{Group: 0, Binding: 0, Space: Uniform, Name: "params", Type: Named{"Params"}},
			{Group: 0, Binding: 1, Space: Storage, Access: Read, Name: "src", Type: Array{Elem: Vec{4, F32}}},
			{Group: 0, Binding: 2, Space: Storage, Access: ReadWrite, Name: "dst", Type: Array{Elem: Vec{4, F32}}},
		},
		Kernels: []Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: Vec{3, U32}}},
			Body: []Stmt{
				Let{"i", Member{Name{"gid"}, "x"}},
				If{
					Cond: Binary{">=", Name{"i"}, Call{"arrayLength", []Expr{AddrOf{Name{"src"}}}}},
					Then: []Stmt{Return{}},
				},
				Assign{
					Target: Index{Name{"dst"}, Name{"i"}},
					Value: Binary{"+",
						Binary{"*", Index{Name{"src"}, Name{"i"}}, Member{Name{"params"}, "scale"}},
						Member{Name{"params"}, "bias"}},
				},
			},
		}},
	}
}
