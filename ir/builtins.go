package ir

// VecConstructor reports whether name is a vector constructor call —
// vec2/vec3/vec4 with an optional element suffix (`u` → u32, `i` → i32,
// `f` or none → f32) — and returns the component count and element scalar.
//
// Vector construction (`vec3(x, y, z)`) is the primitive that lets imperative
// kernels assemble new vectors rather than only read/swizzle existing ones —
// required for any real simulation kernel (particle integration, force fields,
// procedural geometry). Each backend spells it through its own type dialect
// (WGSL `vec3<f32>(…)`, GLSL `vec3(…)`, Metal `float3(…)`); the CPU interpreter
// builds the component slice directly.
func VecConstructor(name string) (n int, elem Scalar, ok bool) {
	if len(name) < 4 || name[0:3] != "vec" {
		return 0, Scalar{}, false
	}
	switch name[3] {
	case '2':
		n = 2
	case '3':
		n = 3
	case '4':
		n = 4
	default:
		return 0, Scalar{}, false
	}
	switch name[4:] {
	case "", "f":
		elem = F32
	case "u":
		elem = U32
	case "i":
		elem = I32
	default:
		return 0, Scalar{}, false
	}
	return n, elem, true
}

// ScalarCast reports whether name is a scalar conversion — f32/i32/u32(x) — and
// returns the target scalar type. Casts are required to mix integer loop
// counters and indices with float uniforms (e.g. `f32(fi) < forceCount`), the
// pattern every dynamic-count compute loop uses. Each backend spells the
// conversion in its own dialect (WGSL `f32`, GLSL/Metal `float`/`int`/`uint`).
func ScalarCast(name string) (Scalar, bool) {
	switch name {
	case "f32":
		return F32, true
	case "i32":
		return I32, true
	case "u32":
		return U32, true
	}
	return Scalar{}, false
}
