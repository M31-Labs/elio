package conformance

import (
	"strings"
	"testing"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
)

// TestVectorConstructorSpelling proves the vector-constructor primitive lowers
// to each backend's native spelling: WGSL `vec3<f32>(…)`, GLSL `vec3(…)`,
// Metal `float3(…)`, and the unsigned variant to `vec3<u32>` / `uvec3` /
// `uint3`. Construction is the primitive that unblocks real simulation kernels
// (e.g. the galaxy particle integrator) which must assemble new vectors.
func TestVectorConstructorSpelling(t *testing.T) {
	lit := func(s string) ir.Expr { return ir.Lit{Text: s} }
	mod := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "acc", Type: ir.Array{Elem: ir.F32}}},
		Kernels: []ir.Kernel{{
			Name: "main", WorkgroupSize: [3]int{1, 1, 1},
			Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "v", Value: ir.Call{Func: "vec3", Args: []ir.Expr{lit("1.0"), lit("2.0"), lit("3.0")}}},
				ir.Let{Name: "iv", Value: ir.Call{Func: "vec3u", Args: []ir.Expr{lit("1u"), lit("2u"), lit("3u")}}},
				ir.Assign{
					Target: ir.Index{E: ir.Name{Name: "acc"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
					Value:  ir.Member{E: ir.Name{Name: "v"}, Field: "y"},
				},
			},
		}},
	}

	wsrc, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}
	if !strings.Contains(wsrc, "vec3<f32>(1.0, 2.0, 3.0)") {
		t.Errorf("WGSL missing native vec3<f32> constructor:\n%s", wsrc)
	}
	if !strings.Contains(wsrc, "vec3<u32>(") {
		t.Errorf("WGSL missing native vec3<u32> constructor:\n%s", wsrc)
	}

	gsrc, err := glsl.Emit(mod)
	if err != nil {
		t.Fatalf("glsl.Emit: %v", err)
	}
	if !strings.Contains(gsrc, "vec3(1.0, 2.0, 3.0)") {
		t.Errorf("GLSL missing native vec3 constructor:\n%s", gsrc)
	}
	if !strings.Contains(gsrc, "uvec3(") {
		t.Errorf("GLSL missing native uvec3 constructor:\n%s", gsrc)
	}

	msrc, err := metal.Emit(mod)
	if err != nil {
		t.Fatalf("metal.Emit: %v", err)
	}
	if !strings.Contains(msrc, "float3(1.0, 2.0, 3.0)") {
		t.Errorf("Metal missing native float3 constructor:\n%s", msrc)
	}
	if !strings.Contains(msrc, "uint3(") {
		t.Errorf("Metal missing native uint3 constructor:\n%s", msrc)
	}
}
