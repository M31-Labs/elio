package sema

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
)

func gidX() ir.Expr { return ir.Member{E: ir.Name{Name: "gid"}, Field: "x"} }

// gatherKernel builds a one-kernel module over the given bindings and body.
func gatherKernel(bindings []ir.Binding, body ...ir.Stmt) *ir.Module {
	return &ir.Module{
		Bindings: bindings,
		Kernels: []ir.Kernel{{
			Name:          "k",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body:          body,
		}},
	}
}

func rwBuf(slot int, name string, elem ir.Type) ir.Binding {
	return ir.Binding{Group: 0, Binding: slot, Space: ir.Storage, Access: ir.ReadWrite, Name: name, Type: ir.Array{Elem: elem}}
}

func roBuf(slot int, name string, elem ir.Type) ir.Binding {
	return ir.Binding{Group: 0, Binding: slot, Space: ir.Storage, Access: ir.Read, Name: name, Type: ir.Array{Elem: elem}}
}

func hasDiag(errs []error, subs ...string) bool {
	for _, e := range errs {
		ok := true
		for _, s := range subs {
			if !strings.Contains(e.Error(), s) {
				ok = false
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func TestCheckGatherScaleBiasPasses(t *testing.T) {
	if errs := CheckGather(ir.ScaleBias(), "main"); len(errs) != 0 {
		t.Fatalf("ScaleBias should be gather-compatible: %v", errs)
	}
}

func TestCheckGatherUnknownKernel(t *testing.T) {
	if errs := CheckGather(ir.ScaleBias(), "nope"); len(errs) == 0 {
		t.Fatal("expected an error for a missing kernel")
	}
}

func TestCheckGatherWriteForm(t *testing.T) {
	bind := []ir.Binding{rwBuf(0, "dst", ir.F32)}
	cases := []struct {
		name string
		body []ir.Stmt
		bad  bool
	}{
		{"direct gid.x", []ir.Stmt{ir.Assign{Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()}, Value: ir.Lit{Text: "1.0"}}}, false},
		{"let alias", []ir.Stmt{
			ir.Let{Name: "i", Value: gidX()},
			ir.Assign{Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: ir.Name{Name: "i"}}, Value: ir.Lit{Text: "1.0"}},
		}, false},
		{"offset index", []ir.Stmt{
			ir.Let{Name: "i", Value: gidX()},
			ir.Assign{Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: ir.Binary{Op: "+", L: ir.Name{Name: "i"}, R: ir.Lit{Text: "1u"}}}, Value: ir.Lit{Text: "1.0"}},
		}, true},
		{"gid.y index", []ir.Stmt{ir.Assign{Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: ir.Member{E: ir.Name{Name: "gid"}, Field: "y"}}, Value: ir.Lit{Text: "1.0"}}}, true},
		{"compound", []ir.Stmt{ir.Assign{Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()}, Op: "+", Value: ir.Lit{Text: "1.0"}}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := CheckGather(gatherKernel(bind, tc.body...), "k")
			if tc.bad && !hasDiag(errs, "rule 3") {
				t.Errorf("want a rule 3 diagnostic, got %v", errs)
			}
			if !tc.bad && len(errs) != 0 {
				t.Errorf("want no diagnostics, got %v", errs)
			}
		})
	}
}

func TestCheckGatherFieldWriteRejected(t *testing.T) {
	bind := []ir.Binding{rwBuf(0, "dst", ir.Vec{N: 4, Elem: ir.F32})}
	m := gatherKernel(bind, ir.Assign{
		Target: ir.Member{E: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()}, Field: "x"},
		Value:  ir.Lit{Text: "1.0"},
	})
	if errs := CheckGather(m, "k"); !hasDiag(errs, "rule 3") {
		t.Errorf("swizzle write into a buffer element should fail rule 3: %v", errs)
	}
}

func TestCheckGatherFiveOutputs(t *testing.T) {
	var bind []ir.Binding
	var body []ir.Stmt
	for i, n := range []string{"a", "b", "c", "d", "e"} {
		bind = append(bind, rwBuf(i, n, ir.F32))
		body = append(body, ir.Assign{Target: ir.Index{E: ir.Name{Name: n}, Idx: gidX()}, Value: ir.Lit{Text: "1.0"}})
	}
	if errs := CheckGather(gatherKernel(bind, body...), "k"); !hasDiag(errs, "rule 4", "five") && !hasDiag(errs, "rule 4", "5") {
		t.Errorf("want a rule 4 diagnostic for five outputs: %v", errs)
	}
	// Four buffers satisfy rule 4 but cost 64 bytes per pixel (rule 8).
	errs := CheckGather(gatherKernel(bind[:4], body[:4]...), "k")
	if hasDiag(errs, "rule 4") || !hasDiag(errs, "rule 8", "64 bytes per pixel") {
		t.Errorf("four buffers should fail only the byte cap: %v", errs)
	}
	if errs := CheckGather(gatherKernel(bind[:2], body[:2]...), "k"); len(errs) != 0 {
		t.Errorf("two outputs should pass: %v", errs)
	}
}

func TestCheckGatherReadWrittenBuffer(t *testing.T) {
	bind := []ir.Binding{rwBuf(0, "dst", ir.F32)}
	m := gatherKernel(bind, ir.Assign{
		Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()},
		Value:  ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()},
	})
	if errs := CheckGather(m, "k"); !hasDiag(errs, "rule 4", "dst") {
		t.Errorf("reading a written buffer should fail rule 4: %v", errs)
	}
}

func TestCheckGatherBufferElementTypes(t *testing.T) {
	structs := []ir.Struct{{Name: "S", Fields: []ir.Field{{Name: "a", Type: ir.F32}}}}
	// struct buffer
	m := gatherKernel([]ir.Binding{
		roBuf(0, "src", ir.Named{Name: "S"}),
		rwBuf(1, "dst", ir.F32),
	}, ir.Assign{
		Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()},
		Value:  ir.Member{E: ir.Index{E: ir.Name{Name: "src"}, Idx: gidX()}, Field: "a"},
	})
	m.Structs = structs
	if errs := CheckGather(m, "k"); !hasDiag(errs, "rule 5", "src") {
		t.Errorf("struct buffer should fail rule 5: %v", errs)
	}
	// vec3 buffer
	m = gatherKernel([]ir.Binding{
		roBuf(0, "src", ir.Vec{N: 3, Elem: ir.F32}),
		rwBuf(1, "dst", ir.F32),
	}, ir.Assign{
		Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()},
		Value:  ir.Member{E: ir.Index{E: ir.Name{Name: "src"}, Idx: gidX()}, Field: "x"},
	})
	if errs := CheckGather(m, "k"); !hasDiag(errs, "rule 5", "src") {
		t.Errorf("vec3 buffer should fail rule 5: %v", errs)
	}
}

func TestCheckGatherAtomics(t *testing.T) {
	m := gatherKernel([]ir.Binding{
		{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "counts", Type: ir.Array{Elem: ir.Atomic{Elem: ir.U32}, Len: 4}},
	}, ir.Do{Expr: ir.Call{Func: "atomicAdd", Args: []ir.Expr{
		ir.AddrOf{E: ir.Index{E: ir.Name{Name: "counts"}, Idx: ir.Lit{Text: "0"}}}, ir.Lit{Text: "1u"}}}})
	if errs := CheckGather(m, "k"); !hasDiag(errs, "rule 1", "atomic") {
		t.Errorf("atomics should fail rule 1: %v", errs)
	}
}

func TestCheckGatherUniformFields(t *testing.T) {
	mk := func(f ir.Type) *ir.Module {
		m := gatherKernel([]ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "p", Type: ir.Named{Name: "P"}},
			rwBuf(1, "dst", ir.F32),
		}, ir.Assign{
			Target: ir.Index{E: ir.Name{Name: "dst"}, Idx: gidX()},
			Value:  ir.Lit{Text: "1.0"},
		}, ir.Do{Expr: ir.Name{Name: "p"}})
		m.Structs = []ir.Struct{{Name: "P", Fields: []ir.Field{{Name: "f", Type: f}}}}
		return m
	}
	for _, ok := range []ir.Type{ir.F32, ir.U32, ir.Vec{N: 3, Elem: ir.F32}, ir.Vec{N: 4, Elem: ir.U32}} {
		if errs := CheckGather(mk(ok), "k"); len(errs) != 0 {
			t.Errorf("uniform field %v should pass: %v", ok, errs)
		}
	}
	for _, bad := range []ir.Type{ir.Mat{Cols: 4, Rows: 4, Elem: ir.F32}, ir.Array{Elem: ir.Vec{N: 4, Elem: ir.F32}, Len: 6}, ir.Bool} {
		if errs := CheckGather(mk(bad), "k"); !hasDiag(errs, "rule 6") {
			t.Errorf("uniform field %v should fail rule 6: %v", bad, errs)
		}
	}
}
