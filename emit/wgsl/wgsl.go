// Package wgsl lowers an Elio compute ir.Module to WGSL compute source — the
// first Elio backend. It targets WebGPU (browser + GoSX desktop), and is the
// reference the Metal / SPIR-V / CPU-fallback backends will mirror. The output
// is what examples/cull hand-writes today, generated from the IR instead.
package wgsl

import (
	"fmt"
	"strings"

	"m31labs.dev/elio/ir"
)

// Emit renders m as WGSL compute source: struct declarations, module bindings,
// then each kernel.
func Emit(m *ir.Module) (string, error) {
	var b strings.Builder
	for _, s := range m.Structs {
		emitStruct(&b, s)
		b.WriteString("\n")
	}
	for _, bnd := range m.Bindings {
		emitBinding(&b, bnd)
	}
	if len(m.Bindings) > 0 {
		b.WriteString("\n")
	}
	shared := false
	for _, k := range m.Kernels {
		for _, sh := range k.Shared {
			fmt.Fprintf(&b, "var<workgroup> %s : %s;\n", sh.Name, typeName(sh.Type))
			shared = true
		}
	}
	if shared {
		b.WriteString("\n")
	}
	for i, k := range m.Kernels {
		if i > 0 {
			b.WriteString("\n")
		}
		if err := emitKernel(&b, k); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

func emitStruct(b *strings.Builder, s ir.Struct) {
	fmt.Fprintf(b, "struct %s {\n", s.Name)
	for _, f := range s.Fields {
		fmt.Fprintf(b, "  %s : %s,\n", f.Name, typeName(f.Type))
	}
	b.WriteString("};\n")
}

func emitBinding(b *strings.Builder, bnd ir.Binding) {
	qual := string(bnd.Space)
	if bnd.Space == ir.Storage {
		qual = fmt.Sprintf("storage, %s", bnd.Access)
	}
	fmt.Fprintf(b, "@group(%d) @binding(%d) var<%s> %s : %s;\n",
		bnd.Group, bnd.Binding, qual, bnd.Name, typeName(bnd.Type))
}

func emitKernel(b *strings.Builder, k ir.Kernel) error {
	fmt.Fprintf(b, "@compute @workgroup_size(%s)\n", workgroup(k.WorkgroupSize))
	params := make([]string, len(k.Builtins))
	for i, bi := range k.Builtins {
		params[i] = fmt.Sprintf("@builtin(%s) %s : %s", bi.Builtin, bi.Name, typeName(bi.Type))
	}
	fmt.Fprintf(b, "fn %s(%s) {\n", k.Name, strings.Join(params, ", "))
	for _, s := range k.Body {
		if err := emitStmt(b, s, 1); err != nil {
			return err
		}
	}
	b.WriteString("}\n")
	return nil
}

func workgroup(s [3]int) string {
	if s[1] <= 1 && s[2] <= 1 {
		return fmt.Sprintf("%d", s[0])
	}
	return fmt.Sprintf("%d, %d, %d", s[0], s[1], s[2])
}

func emitStmt(b *strings.Builder, s ir.Stmt, depth int) error {
	pad := strings.Repeat("  ", depth)
	switch x := s.(type) {
	case ir.Let:
		fmt.Fprintf(b, "%slet %s = %s;\n", pad, x.Name, expr(x.Value))
	case ir.Var:
		fmt.Fprintf(b, "%s%s;\n", pad, varDecl(x))
	case ir.Assign:
		fmt.Fprintf(b, "%s%s = %s;\n", pad, expr(x.Target), expr(x.Value))
	case ir.Return:
		fmt.Fprintf(b, "%sreturn;\n", pad)
	case ir.Break:
		fmt.Fprintf(b, "%sbreak;\n", pad)
	case ir.Barrier:
		fmt.Fprintf(b, "%sworkgroupBarrier();\n", pad)
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
	default:
		return fmt.Errorf("wgsl: unsupported statement %T", s)
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

// stmtInline renders a for-header init/post statement without indentation or a
// trailing semicolon.
func stmtInline(s ir.Stmt) string {
	switch x := s.(type) {
	case ir.Var:
		return varDecl(x)
	case ir.Assign:
		return fmt.Sprintf("%s = %s", expr(x.Target), expr(x.Value))
	case ir.Let:
		return fmt.Sprintf("let %s = %s", x.Name, expr(x.Value))
	}
	return ""
}

func varDecl(x ir.Var) string {
	if x.Type != nil {
		return fmt.Sprintf("var %s : %s = %s", x.Name, typeName(x.Type), expr(x.Init))
	}
	return fmt.Sprintf("var %s = %s", x.Name, expr(x.Init))
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
		args := make([]string, len(x.Args))
		for i, a := range x.Args {
			args[i] = expr(a)
		}
		return fmt.Sprintf("%s(%s)", x.Func, strings.Join(args, ", "))
	case ir.Index:
		return fmt.Sprintf("%s[%s]", expr(x.E), expr(x.Idx))
	case ir.Member:
		return fmt.Sprintf("%s.%s", expr(x.E), x.Field)
	case ir.AddrOf:
		return "&" + expr(x.E)
	}
	return "/* unknown expr */"
}

func typeName(t ir.Type) string {
	switch x := t.(type) {
	case ir.Scalar:
		return x.Name
	case ir.Vec:
		return fmt.Sprintf("vec%d<%s>", x.N, x.Elem.Name)
	case ir.Mat:
		return fmt.Sprintf("mat%dx%d<%s>", x.Cols, x.Rows, x.Elem.Name)
	case ir.Array:
		if x.Len == 0 {
			return fmt.Sprintf("array<%s>", typeName(x.Elem))
		}
		return fmt.Sprintf("array<%s, %d>", typeName(x.Elem), x.Len)
	case ir.Atomic:
		return fmt.Sprintf("atomic<%s>", x.Elem.Name)
	case ir.Named:
		return x.Name
	}
	return "/* unknown type */"
}
