package sema

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

// TestValidSamplesPass pins that the two reference kernels are accepted clean.
func TestValidSamplesPass(t *testing.T) {
	for _, tc := range []struct {
		name string
		mod  *ir.Module
	}{
		{"cull", ir.CullKernel()},
		{"scalebias", ir.ScaleBias()},
		{"reduce", ir.WorkgroupReduce()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Check(tc.mod); len(errs) != 0 {
				t.Fatalf("expected no diagnostics, got:\n%v", Errors(errs))
			}
		})
	}
}

// TestTypeChecks covers the type-aware layer: struct field existence, vector
// swizzle validity, and indexing non-indexable types.
func TestTypeChecks(t *testing.T) {
	// A module with a Params struct (a vec4 + a vec2) bound as a uniform, plus a
	// vec3 builtin, so member/swizzle/index checks have concrete types to see.
	mod := func(body ...ir.Stmt) *ir.Module {
		return &ir.Module{
			Structs: []ir.Struct{{Name: "Params", Fields: []ir.Field{
				{Name: "tint", Type: ir.Vec{N: 4, Elem: ir.F32}},
				{Name: "uv", Type: ir.Vec{N: 2, Elem: ir.F32}},
			}}},
			Bindings: []ir.Binding{
				{Group: 0, Binding: 0, Space: ir.Uniform, Name: "p", Type: ir.Named{Name: "Params"}},
				{Group: 0, Binding: 1, Space: ir.Uniform, Name: "scale", Type: ir.F32},
			},
			Kernels: []ir.Kernel{{
				Name: "main", WorkgroupSize: [3]int{64, 1, 1},
				Builtins: []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
				Body:     body,
			}},
		}
	}
	cases := []struct {
		name string
		body []ir.Stmt
		want string
	}{
		{"good field+swizzle", []ir.Stmt{
			ir.Let{Name: "a", Value: ir.Member{E: ir.Member{E: ir.Name{Name: "p"}, Field: "tint"}, Field: "xyz"}},
		}, ""},
		{"bad struct field", []ir.Stmt{
			ir.Let{Name: "a", Value: ir.Member{E: ir.Name{Name: "p"}, Field: "tnit"}},
		}, `struct "Params" has no field "tnit"`},
		{"swizzle out of range", []ir.Stmt{
			// gid is vec3; .w (component 3) is out of range
			ir.Let{Name: "a", Value: ir.Member{E: ir.Name{Name: "gid"}, Field: "w"}},
		}, `invalid swizzle "w" on vec3u`},
		{"swizzle on vec2 ok", []ir.Stmt{
			ir.Let{Name: "a", Value: ir.Member{E: ir.Member{E: ir.Name{Name: "p"}, Field: "uv"}, Field: "yx"}},
		}, ""},
		{"index a scalar", []ir.Stmt{
			ir.Let{Name: "a", Value: ir.Index{E: ir.Name{Name: "scale"}, Idx: ir.Lit{Text: "0"}}},
		}, "cannot index f32"},
		{"field of a scalar", []ir.Stmt{
			ir.Let{Name: "a", Value: ir.Member{E: ir.Name{Name: "scale"}, Field: "x"}},
		}, `f32 has no field "x"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Errors(Check(mod(tc.body...)))
			if tc.want == "" {
				if got != nil {
					t.Fatalf("expected valid, got:\n%v", got)
				}
				return
			}
			if got == nil || !strings.Contains(got.Error(), tc.want) {
				t.Fatalf("expected diagnostic containing %q, got:\n%v", tc.want, got)
			}
		})
	}
}

// TestDiagnosticSpan pins that a statement's source position is carried into the
// diagnostic (line:col), when the IR has one.
func TestDiagnosticSpan(t *testing.T) {
	mod := &ir.Module{Kernels: []ir.Kernel{{
		Name: "main", WorkgroupSize: [3]int{1, 1, 1},
		Body: []ir.Stmt{
			ir.Let{Name: "x", Value: ir.Name{Name: "missing"}, Span: ir.Span{Line: 5, Col: 11}},
		},
	}}}
	got := Errors(Check(mod))
	if got == nil || !strings.Contains(got.Error(), "5:11:") {
		t.Fatalf("expected diagnostic to carry 5:11, got: %v", got)
	}
}

// TestConstSemantics pins that a module constant resolves in kernel code but
// cannot be assigned to.
func TestConstSemantics(t *testing.T) {
	mk := func(body ...ir.Stmt) *ir.Module {
		return &ir.Module{
			Consts: []ir.Const{{Name: "PI", Type: ir.F32, Value: ir.Lit{Text: "3.14"}}},
			Kernels: []ir.Kernel{{
				Name: "main", WorkgroupSize: [3]int{1, 1, 1},
				Body: body,
			}},
		}
	}
	// referencing a const is fine
	if errs := Check(mk(ir.Let{Name: "x", Value: ir.Name{Name: "PI"}})); len(errs) != 0 {
		t.Fatalf("expected const reference to be valid, got: %v", Errors(errs))
	}
	// assigning to a const is an error
	got := Errors(Check(mk(ir.Assign{Target: ir.Name{Name: "PI"}, Value: ir.Lit{Text: "1.0"}})))
	if got == nil || !strings.Contains(got.Error(), `cannot assign to "PI"`) {
		t.Fatalf("expected const-immutability diagnostic, got: %v", got)
	}
}

// kernel wraps a body in a minimal module with the given bindings for testing.
func kernel(bindings []ir.Binding, body ...ir.Stmt) *ir.Module {
	return &ir.Module{
		Bindings: bindings,
		Kernels: []ir.Kernel{{
			Name:          "main",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body:          body,
		}},
	}
}

func TestDiagnostics(t *testing.T) {
	rw := ir.Binding{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "dst", Type: ir.Array{Elem: ir.F32}}
	ro := ir.Binding{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Array{Elem: ir.F32}}
	uni := ir.Binding{Group: 0, Binding: 2, Space: ir.Uniform, Name: "cfg", Type: ir.F32}

	cases := []struct {
		name string
		mod  *ir.Module
		want string // substring expected in the joined diagnostics
	}{
		{
			"undefined name",
			kernel(nil, ir.Let{Name: "x", Value: ir.Name{Name: "missing"}}),
			`undefined name "missing"`,
		},
		{
			"assign to let",
			kernel(nil,
				ir.Let{Name: "x", Value: ir.Lit{Text: "0"}},
				ir.Assign{Target: ir.Name{Name: "x"}, Value: ir.Lit{Text: "1"}}),
			`cannot assign to "x"`,
		},
		{
			"assign to read-only storage",
			kernel([]ir.Binding{ro},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "src"}, Idx: ir.Lit{Text: "0"}}, Value: ir.Lit{Text: "1"}}),
			`cannot assign to "src"`,
		},
		{
			"assign to uniform",
			kernel([]ir.Binding{uni},
				ir.Assign{Target: ir.Name{Name: "cfg"}, Value: ir.Lit{Text: "1"}}),
			`cannot assign to "cfg"`,
		},
		{
			"write to read_write storage is ok via index",
			kernel([]ir.Binding{rw},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: ir.Lit{Text: "0"}}, Value: ir.Lit{Text: "1"}}),
			"", // valid
		},
		{
			"addr-of a local",
			kernel(nil,
				ir.Var{Name: "x", Init: ir.Lit{Text: "0"}},
				ir.Let{Name: "p", Value: ir.AddrOf{E: ir.Name{Name: "x"}}}),
			`cannot take the address of "x"`,
		},
		{
			"shared var is assignable; addr-of shared is not",
			&ir.Module{Kernels: []ir.Kernel{{
				Name: "main", WorkgroupSize: [3]int{64, 1, 1},
				Builtins: []ir.Builtin{{Name: "lid", Builtin: "local_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
				Shared:   []ir.Shared{{Name: "tile", Type: ir.Array{Elem: ir.F32, Len: 64}}},
				Body: []ir.Stmt{
					ir.Assign{Target: ir.Index{E: ir.Name{Name: "tile"}, Idx: ir.Member{E: ir.Name{Name: "lid"}, Field: "x"}}, Value: ir.Lit{Text: "0"}},
					ir.Let{Name: "p", Value: ir.AddrOf{E: ir.Name{Name: "tile"}}},
				},
			}}},
			`cannot take the address of "tile"`,
		},
		{
			"addr-of storage is ok",
			kernel([]ir.Binding{ro},
				ir.Let{Name: "n", Value: ir.Call{Func: "arrayLength", Args: []ir.Expr{ir.AddrOf{E: ir.Name{Name: "src"}}}}}),
			"",
		},
		{
			"duplicate binding slot",
			&ir.Module{Bindings: []ir.Binding{
				{Group: 0, Binding: 0, Space: ir.Uniform, Name: "a", Type: ir.F32},
				{Group: 0, Binding: 0, Space: ir.Uniform, Name: "b", Type: ir.F32},
			}},
			"reuses @group(0) @binding(0)",
		},
		{
			"unknown named type",
			&ir.Module{Bindings: []ir.Binding{
				{Group: 0, Binding: 0, Space: ir.Uniform, Name: "cfg", Type: ir.Named{Name: "Nope"}},
			}},
			`unknown type "Nope"`,
		},
		{
			"call arity",
			kernel(nil, ir.Let{Name: "d", Value: ir.Call{Func: "dot", Args: []ir.Expr{ir.Lit{Text: "1"}}}}),
			"dot expects 2 argument(s), got 1",
		},
		{
			"redeclared local",
			kernel(nil,
				ir.Let{Name: "x", Value: ir.Lit{Text: "0"}},
				ir.Let{Name: "x", Value: ir.Lit{Text: "1"}}),
			`"x" is already declared in this block`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Errors(Check(tc.mod))
			if tc.want == "" {
				if got != nil {
					t.Fatalf("expected valid, got:\n%v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("expected diagnostic containing %q, got none", tc.want)
			}
			if !strings.Contains(got.Error(), tc.want) {
				t.Fatalf("expected diagnostic containing %q, got:\n%v", tc.want, got)
			}
		})
	}
}
