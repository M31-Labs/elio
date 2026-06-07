// Package metal lowers an Elio compute ir.Module to Metal Shading Language —
// the second Elio backend (iOS / macOS, the GoSX-native mobile surface).
//
// Metal differs from WGSL in three ways this emitter handles: bindings become
// typed kernel arguments with [[buffer(i)]] attributes (not module globals);
// storage buffers are pointers (device / const device); and MSL has no
// arrayLength, so for every runtime array used in arrayLength(&x) the emitter
// synthesizes a `constant uint& xLength` argument and rewrites the call to it.
package metal

import (
	"fmt"
	"sort"
	"strings"

	"m31labs.dev/elio/emit/internal/prismtypes"
	"m31labs.dev/elio/ir"
	"m31labs.dev/prism/dialect"
)

var metalDialect = dialect.Metal{}

// Emit renders m as a Metal compute source string.
func Emit(m *ir.Module) (string, error) {
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	for _, s := range m.Structs {
		emitStruct(&b, s)
		b.WriteString("\n")
	}
	for _, cn := range m.Consts {
		fmt.Fprintf(&b, "constant %s %s = %s;\n", typeName(cn.Type), cn.Name, expr(cn.Value))
	}
	if len(m.Consts) > 0 {
		b.WriteString("\n")
	}
	for i, k := range m.Kernels {
		if i > 0 {
			b.WriteString("\n")
		}
		if err := emitKernel(&b, m, k); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

func emitStruct(b *strings.Builder, s ir.Struct) {
	fmt.Fprintf(b, "struct %s {\n", s.Name)
	for _, f := range s.Fields {
		// MSL puts the array size after the field name: float4 planes[6];
		if arr, ok := f.Type.(ir.Array); ok && arr.Len > 0 {
			fmt.Fprintf(b, "  %s %s[%d];\n", typeName(arr.Elem), f.Name, arr.Len)
			continue
		}
		fmt.Fprintf(b, "  %s %s;\n", typeName(f.Type), f.Name)
	}
	b.WriteString("};\n")
}

func emitKernel(b *strings.Builder, m *ir.Module, k ir.Kernel) error {
	lengths := lengthParams(k.Body)

	var args []string
	buf := 0
	for _, bnd := range m.Bindings {
		args = append(args, bindingArg(bnd, buf))
		buf++
	}
	for _, name := range lengths {
		args = append(args, fmt.Sprintf("constant uint& %sLength [[buffer(%d)]]", name, buf))
		buf++
	}
	for _, bi := range k.Builtins {
		args = append(args, fmt.Sprintf("%s %s [[%s]]", typeName(bi.Type), bi.Name, metalBuiltin(bi.Builtin)))
	}

	fmt.Fprintf(b, "kernel void %s(\n    %s) {\n", k.Name, strings.Join(args, ",\n    "))
	// Metal threadgroup memory is declared function-local, as the first body
	// statements (WGSL/GLSL hoist the equivalent to module scope).
	for _, sh := range k.Shared {
		if arr, ok := sh.Type.(ir.Array); ok && arr.Len > 0 {
			fmt.Fprintf(b, "  threadgroup %s %s[%d];\n", typeName(arr.Elem), sh.Name, arr.Len)
		} else {
			fmt.Fprintf(b, "  threadgroup %s %s;\n", typeName(sh.Type), sh.Name)
		}
	}
	for _, s := range k.Body {
		if err := emitStmt(b, s, 1); err != nil {
			return err
		}
	}
	b.WriteString("}\n")
	return nil
}

func bindingArg(bnd ir.Binding, buf int) string {
	switch bnd.Space {
	case ir.Uniform:
		return fmt.Sprintf("constant %s& %s [[buffer(%d)]]", typeName(bnd.Type), bnd.Name, buf)
	case ir.Storage:
		elem := bnd.Type
		if arr, ok := bnd.Type.(ir.Array); ok {
			elem = arr.Elem
		}
		qual := "device"
		if bnd.Access == ir.Read {
			qual = "const device"
		}
		return fmt.Sprintf("%s %s* %s [[buffer(%d)]]", qual, typeName(elem), bnd.Name, buf)
	}
	return ""
}

func metalBuiltin(name string) string {
	switch name {
	case "global_invocation_id":
		return "thread_position_in_grid"
	case "local_invocation_id":
		return "thread_position_in_threadgroup"
	case "workgroup_id":
		return "threadgroup_position_in_grid"
	}
	return name
}

// lengthParams returns, in stable order, the names of runtime arrays referenced
// by arrayLength(&name) anywhere in body — each needs a synthesized length arg.
func lengthParams(body []ir.Stmt) []string {
	set := map[string]bool{}
	var walkE func(e ir.Expr)
	walkE = func(e ir.Expr) {
		switch x := e.(type) {
		case ir.Call:
			if x.Func == "arrayLength" && len(x.Args) == 1 {
				if a, ok := x.Args[0].(ir.AddrOf); ok {
					if n, ok := a.E.(ir.Name); ok {
						set[n.Name] = true
					}
				}
			}
			for _, a := range x.Args {
				walkE(a)
			}
		case ir.Binary:
			walkE(x.L)
			walkE(x.R)
		case ir.Unary:
			walkE(x.E)
		case ir.Index:
			walkE(x.E)
			walkE(x.Idx)
		case ir.Member:
			walkE(x.E)
		case ir.AddrOf:
			walkE(x.E)
		}
	}
	var walkS func(s ir.Stmt)
	walkBlock := func(ss []ir.Stmt) {
		for _, s := range ss {
			walkS(s)
		}
	}
	walkS = func(s ir.Stmt) {
		switch x := s.(type) {
		case ir.Let:
			walkE(x.Value)
		case ir.Var:
			walkE(x.Init)
		case ir.Assign:
			walkE(x.Target)
			walkE(x.Value)
		case ir.Do:
			walkE(x.Expr)
		case ir.If:
			walkE(x.Cond)
			walkBlock(x.Then)
			walkBlock(x.Else)
		case ir.For:
			if x.Init != nil {
				walkS(x.Init)
			}
			walkE(x.Cond)
			if x.Post != nil {
				walkS(x.Post)
			}
			walkBlock(x.Body)
		}
	}
	walkBlock(body)
	out := make([]string, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

func emitStmt(b *strings.Builder, s ir.Stmt, depth int) error {
	pad := strings.Repeat("  ", depth)
	switch x := s.(type) {
	case ir.Let:
		fmt.Fprintf(b, "%sauto %s = %s;\n", pad, x.Name, expr(x.Value))
	case ir.Var:
		fmt.Fprintf(b, "%s%s;\n", pad, varDecl(x))
	case ir.Assign:
		fmt.Fprintf(b, "%s%s %s= %s;\n", pad, expr(x.Target), x.Op, expr(x.Value))
	case ir.Return:
		fmt.Fprintf(b, "%sreturn;\n", pad)
	case ir.Break:
		fmt.Fprintf(b, "%sbreak;\n", pad)
	case ir.Barrier:
		fmt.Fprintf(b, "%sthreadgroup_barrier(mem_flags::mem_threadgroup);\n", pad)
	case ir.Do:
		fmt.Fprintf(b, "%s%s;\n", pad, expr(x.Expr))
	case ir.If:
		fmt.Fprintf(b, "%sif (%s) {\n", pad, expr(x.Cond))
		if err := emitBlock(b, x.Then, depth+1); err != nil {
			return err
		}
		if len(x.Else) > 0 {
			fmt.Fprintf(b, "%s} else {\n", pad)
			if err := emitBlock(b, x.Else, depth+1); err != nil {
				return err
			}
		}
		fmt.Fprintf(b, "%s}\n", pad)
	case ir.For:
		fmt.Fprintf(b, "%sfor (%s; %s; %s) {\n", pad, stmtInline(x.Init), expr(x.Cond), stmtInline(x.Post))
		if err := emitBlock(b, x.Body, depth+1); err != nil {
			return err
		}
		fmt.Fprintf(b, "%s}\n", pad)
	case ir.While:
		fmt.Fprintf(b, "%swhile (%s) {\n", pad, expr(x.Cond))
		if err := emitBlock(b, x.Body, depth+1); err != nil {
			return err
		}
		fmt.Fprintf(b, "%s}\n", pad)
	default:
		return fmt.Errorf("metal: unsupported statement %T", s)
	}
	return nil
}

func emitBlock(b *strings.Builder, body []ir.Stmt, depth int) error {
	for _, s := range body {
		if err := emitStmt(b, s, depth); err != nil {
			return err
		}
	}
	return nil
}

func stmtInline(s ir.Stmt) string {
	switch x := s.(type) {
	case ir.Var:
		return varDecl(x)
	case ir.Assign:
		return fmt.Sprintf("%s %s= %s", expr(x.Target), x.Op, expr(x.Value))
	case ir.Let:
		return fmt.Sprintf("auto %s = %s", x.Name, expr(x.Value))
	}
	return ""
}

func varDecl(x ir.Var) string {
	t := "auto"
	if x.Type != nil {
		t = typeName(x.Type)
	}
	return fmt.Sprintf("%s %s = %s", t, x.Name, expr(x.Init))
}

func expr(e ir.Expr) string {
	switch x := e.(type) {
	case ir.Name:
		return x.Name
	case ir.Lit:
		return x.Text
	case ir.Binary:
		return fmt.Sprintf("(%s %s %s)", expr(x.L), x.Op, expr(x.R))
	case ir.Unary:
		return x.Op + expr(x.E)
	case ir.Call:
		return call(x)
	case ir.Index:
		return fmt.Sprintf("%s[%s]", expr(x.E), expr(x.Idx))
	case ir.Member:
		return fmt.Sprintf("%s.%s", expr(x.E), x.Field)
	case ir.AddrOf:
		return "&" + expr(x.E)
	}
	return "/* unknown expr */"
}

func call(c ir.Call) string {
	switch c.Func {
	case "arrayLength":
		if len(c.Args) == 1 {
			if a, ok := c.Args[0].(ir.AddrOf); ok {
				if n, ok := a.E.(ir.Name); ok {
					return n.Name + "Length"
				}
			}
		}
	case "atomicAdd":
		if len(c.Args) == 2 {
			return fmt.Sprintf("atomic_fetch_add_explicit(%s, %s, memory_order_relaxed)", expr(c.Args[0]), expr(c.Args[1]))
		}
	}
	fn := c.Func
	if n, elem, ok := ir.VecConstructor(c.Func); ok {
		fn = typeName(ir.Vec{N: n, Elem: elem}) // float3 / uint3 / int3
	}
	args := make([]string, len(c.Args))
	for i, a := range c.Args {
		args[i] = expr(a)
	}
	return fmt.Sprintf("%s(%s)", fn, strings.Join(args, ", "))
}

func typeName(t ir.Type) string {
	if gt, ok := prismtypes.FromIR(t); ok {
		return metalDialect.TypeName(gt)
	}
	// Atomic, Named, and Array-of-those are not modelled by prism/gputype —
	// spell them locally.
	// Metal atomics are atomic_int / atomic_uint (MSL standard library types).
	switch x := t.(type) {
	case ir.Atomic:
		if x.Elem.Name == "i32" {
			return "atomic_int"
		}
		return "atomic_uint"
	case ir.Named:
		return x.Name
	case ir.Array:
		// Element type is Atomic or Named (prism can't handle it).
		if x.Len == 0 {
			return typeName(x.Elem)
		}
		return fmt.Sprintf("%s[%d]", typeName(x.Elem), x.Len)
	}
	return "/* unknown type */"
}
