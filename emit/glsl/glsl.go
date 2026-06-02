// Package glsl lowers an Elio compute ir.Module to GLSL compute shader source
// (#version 450) — the Vulkan / desktop-GL / GLES-3.1 backend. It is the third
// text backend after WGSL and Metal, and emits SPIR-V-ready source (validated
// with glslangValidator).
//
// GLSL has no type inference (unlike WGSL `let` / MSL `auto`), so this backend
// carries a small inline type inferencer: as each `let` / `var` is emitted, the
// value's type is inferred from the bindings, structs, builtins, and prior
// locals, and the declaration is spelled with the concrete GLSL type. GLSL also
// differs in resource syntax (uniform / buffer blocks, std140 / std430),
// atomics (no &, lvalue-direct), runtime length (`x.length()`), and a set of
// reserved words (input, output, …) that authored names are escaped around.
package glsl

import (
	"fmt"
	"strings"

	"m31labs.dev/elio/ir"
)

// Emit renders m as a #version 450 GLSL compute source string.
func Emit(m *ir.Module) (string, error) {
	e := &emitter{
		m:        m,
		structs:  map[string]ir.Struct{},
		bindings: map[string]ir.Type{},
		glName:   map[string]string{},
		builtins: map[string]ir.Type{},
		inlined:  map[string]bool{},
		locals:   map[string]ir.Type{},
	}
	for _, s := range m.Structs {
		e.structs[s.Name] = s
	}
	for _, b := range m.Bindings {
		e.bindings[b.Name] = b.Type
		if b.Space == ir.Uniform {
			if n, ok := b.Type.(ir.Named); ok {
				e.inlined[n.Name] = true
			}
		}
	}
	return e.emit()
}

type emitter struct {
	m        *ir.Module
	structs  map[string]ir.Struct
	bindings map[string]ir.Type
	glName   map[string]string // builtin param name → gl_ spelling
	builtins map[string]ir.Type
	inlined  map[string]bool // struct names inlined as uniform blocks (no struct decl)
	locals   map[string]ir.Type
	b        strings.Builder
}

func (e *emitter) emit() (string, error) {
	e.b.WriteString("#version 450\n\n")
	for _, s := range e.m.Structs {
		if e.inlined[s.Name] {
			continue // emitted inline as a uniform block at its binding
		}
		e.emitStruct(s)
		e.b.WriteString("\n")
	}
	for _, b := range e.m.Bindings {
		e.emitBinding(b)
	}
	if len(e.m.Bindings) > 0 {
		e.b.WriteString("\n")
	}
	shared := false
	for _, k := range e.m.Kernels {
		for _, sh := range k.Shared {
			if arr, ok := sh.Type.(ir.Array); ok && arr.Len > 0 {
				fmt.Fprintf(&e.b, "shared %s %s[%d];\n", typeName(arr.Elem), safe(sh.Name), arr.Len)
			} else {
				fmt.Fprintf(&e.b, "shared %s %s;\n", typeName(sh.Type), safe(sh.Name))
			}
			shared = true
		}
	}
	if shared {
		e.b.WriteString("\n")
	}
	for i, k := range e.m.Kernels {
		if i > 0 {
			e.b.WriteString("\n")
		}
		if err := e.emitKernel(k); err != nil {
			return "", err
		}
	}
	return e.b.String(), nil
}

func (e *emitter) emitStruct(s ir.Struct) {
	fmt.Fprintf(&e.b, "struct %s {\n", s.Name)
	for _, f := range s.Fields {
		e.b.WriteString("  " + fieldDecl(f) + ";\n")
	}
	e.b.WriteString("};\n")
}

// fieldDecl spells a struct/block field; GLSL puts array size after the name.
func fieldDecl(f ir.Field) string {
	if arr, ok := f.Type.(ir.Array); ok && arr.Len > 0 {
		return fmt.Sprintf("%s %s[%d]", typeName(arr.Elem), safe(f.Name), arr.Len)
	}
	return fmt.Sprintf("%s %s", typeName(f.Type), safe(f.Name))
}

func (e *emitter) emitBinding(b ir.Binding) {
	name := safe(b.Name)
	// Block-type names live in their own namespace and aren't reserved, so use
	// the raw name there; the instance/member name is escaped. This also avoids
	// "__" (raw "_ssbo" suffix on an escaped "name_" would read "name__ssbo").
	switch b.Space {
	case ir.Uniform:
		fmt.Fprintf(&e.b, "layout(std140, binding = %d) uniform %s_ubo {\n", b.Binding, b.Name)
		if n, ok := b.Type.(ir.Named); ok {
			for _, f := range e.structs[n.Name].Fields {
				e.b.WriteString("  " + fieldDecl(f) + ";\n")
			}
		}
		fmt.Fprintf(&e.b, "} %s;\n", name)
	case ir.Storage:
		access := ""
		if b.Access == ir.Read {
			access = "readonly "
		}
		elem := b.Type
		if arr, ok := b.Type.(ir.Array); ok {
			elem = arr.Elem
		}
		fmt.Fprintf(&e.b, "layout(std430, binding = %d) %sbuffer %s_ssbo { %s %s[]; };\n",
			b.Binding, access, b.Name, storageElem(elem), name)
	}
}

func storageElem(t ir.Type) string {
	if a, ok := t.(ir.Atomic); ok {
		return typeName(a.Elem)
	}
	return typeName(t)
}

func (e *emitter) emitKernel(k ir.Kernel) error {
	ws := k.WorkgroupSize
	fmt.Fprintf(&e.b, "layout(local_size_x = %d", ws[0])
	if ws[1] > 1 || ws[2] > 1 {
		fmt.Fprintf(&e.b, ", local_size_y = %d, local_size_z = %d", ws[1], ws[2])
	}
	e.b.WriteString(") in;\n\n")

	for _, bi := range k.Builtins {
		e.builtins[bi.Name] = bi.Type
		e.glName[bi.Name] = glBuiltin(bi.Builtin)
	}
	e.locals = map[string]ir.Type{}

	e.b.WriteString("void main() {\n")
	for _, s := range k.Body {
		if err := e.stmt(s, 1); err != nil {
			return err
		}
	}
	e.b.WriteString("}\n")
	return nil
}

func glBuiltin(name string) string {
	switch name {
	case "global_invocation_id":
		return "gl_GlobalInvocationID"
	case "local_invocation_id":
		return "gl_LocalInvocationID"
	case "workgroup_id":
		return "gl_WorkGroupID"
	}
	return name
}

func (e *emitter) stmt(s ir.Stmt, depth int) error {
	pad := strings.Repeat("  ", depth)
	switch x := s.(type) {
	case ir.Let:
		t := e.infer(x.Value)
		e.locals[x.Name] = t
		fmt.Fprintf(&e.b, "%s%s %s = %s;\n", pad, typeName(t), safe(x.Name), e.expr(x.Value))
	case ir.Var:
		t := x.Type
		if t == nil {
			t = e.infer(x.Init)
		}
		e.locals[x.Name] = t
		fmt.Fprintf(&e.b, "%s%s %s = %s;\n", pad, typeName(t), safe(x.Name), e.expr(x.Init))
	case ir.Assign:
		fmt.Fprintf(&e.b, "%s%s = %s;\n", pad, e.expr(x.Target), e.expr(x.Value))
	case ir.Return:
		fmt.Fprintf(&e.b, "%sreturn;\n", pad)
	case ir.Break:
		fmt.Fprintf(&e.b, "%sbreak;\n", pad)
	case ir.Barrier:
		// Flush shared writes, then synchronize execution.
		fmt.Fprintf(&e.b, "%smemoryBarrierShared();\n%sbarrier();\n", pad, pad)
	case ir.Do:
		fmt.Fprintf(&e.b, "%s%s;\n", pad, e.expr(x.Expr))
	case ir.If:
		fmt.Fprintf(&e.b, "%sif (%s) {\n", pad, e.expr(x.Cond))
		if err := e.block(x.Then, depth+1); err != nil {
			return err
		}
		if len(x.Else) > 0 {
			fmt.Fprintf(&e.b, "%s} else {\n", pad)
			if err := e.block(x.Else, depth+1); err != nil {
				return err
			}
		}
		fmt.Fprintf(&e.b, "%s}\n", pad)
	case ir.For:
		fmt.Fprintf(&e.b, "%sfor (%s; %s; %s) {\n", pad, e.inlineStmt(x.Init), e.expr(x.Cond), e.inlineStmt(x.Post))
		if err := e.block(x.Body, depth+1); err != nil {
			return err
		}
		fmt.Fprintf(&e.b, "%s}\n", pad)
	default:
		return fmt.Errorf("glsl: unsupported statement %T", s)
	}
	return nil
}

func (e *emitter) block(body []ir.Stmt, depth int) error {
	for _, s := range body {
		if err := e.stmt(s, depth); err != nil {
			return err
		}
	}
	return nil
}

func (e *emitter) inlineStmt(s ir.Stmt) string {
	switch x := s.(type) {
	case ir.Var:
		t := x.Type
		if t == nil {
			t = e.infer(x.Init)
		}
		e.locals[x.Name] = t
		return fmt.Sprintf("%s %s = %s", typeName(t), safe(x.Name), e.expr(x.Init))
	case ir.Assign:
		return fmt.Sprintf("%s = %s", e.expr(x.Target), e.expr(x.Value))
	case ir.Let:
		t := e.infer(x.Value)
		e.locals[x.Name] = t
		return fmt.Sprintf("%s %s = %s", typeName(t), safe(x.Name), e.expr(x.Value))
	}
	return ""
}

func (e *emitter) expr(ex ir.Expr) string {
	switch x := ex.(type) {
	case ir.Name:
		if gl, ok := e.glName[x.Name]; ok {
			return gl
		}
		return safe(x.Name)
	case ir.Lit:
		return x.Text
	case ir.Binary:
		return fmt.Sprintf("(%s %s %s)", e.expr(x.L), x.Op, e.expr(x.R))
	case ir.Unary:
		return x.Op + e.expr(x.E)
	case ir.Call:
		return e.call(x)
	case ir.Index:
		return fmt.Sprintf("%s[%s]", e.expr(x.E), e.expr(x.Idx))
	case ir.Member:
		return fmt.Sprintf("%s.%s", e.expr(x.E), safe(x.Field))
	case ir.AddrOf:
		return e.expr(x.E) // GLSL has no address-of; atomics take the lvalue
	}
	return "/* unknown expr */"
}

func (e *emitter) call(c ir.Call) string {
	switch c.Func {
	case "arrayLength":
		if len(c.Args) == 1 {
			if a, ok := c.Args[0].(ir.AddrOf); ok {
				if n, ok := a.E.(ir.Name); ok {
					return safe(n.Name) + ".length()"
				}
			}
		}
	case "atomicAdd":
		if len(c.Args) == 2 {
			return fmt.Sprintf("atomicAdd(%s, %s)", e.expr(c.Args[0]), e.expr(c.Args[1]))
		}
	}
	args := make([]string, len(c.Args))
	for i, a := range c.Args {
		args[i] = e.expr(a)
	}
	return fmt.Sprintf("%s(%s)", c.Func, strings.Join(args, ", "))
}

// --- type inference (GLSL needs concrete decl types) ---

func (e *emitter) infer(ex ir.Expr) ir.Type {
	switch x := ex.(type) {
	case ir.Name:
		if t, ok := e.locals[x.Name]; ok {
			return t
		}
		if t, ok := e.builtins[x.Name]; ok {
			return t
		}
		if t, ok := e.bindings[x.Name]; ok {
			return t
		}
		return ir.F32
	case ir.Lit:
		switch {
		case x.Text == "true" || x.Text == "false":
			return ir.Bool
		case strings.HasSuffix(x.Text, "u"):
			return ir.U32
		case strings.Contains(x.Text, "."):
			return ir.F32
		default:
			return ir.I32
		}
	case ir.Binary:
		switch x.Op {
		case "<", ">", "<=", ">=", "==", "!=":
			return ir.Bool
		}
		lt := e.infer(x.L)
		if isScalar(lt) {
			if rt := e.infer(x.R); !isScalar(rt) {
				return rt // scalar op vector → vector
			}
		}
		return lt
	case ir.Unary:
		if x.Op == "!" {
			return ir.Bool
		}
		return e.infer(x.E)
	case ir.Call:
		switch x.Func {
		case "arrayLength":
			return ir.U32
		case "dot", "length", "distance":
			return ir.F32
		case "atomicAdd":
			return ir.U32
		}
		if len(x.Args) > 0 {
			return e.infer(x.Args[0])
		}
		return ir.F32
	case ir.Index:
		switch b := e.infer(x.E).(type) {
		case ir.Array:
			return b.Elem
		case ir.Mat:
			return ir.Vec{N: b.Rows, Elem: b.Elem}
		case ir.Vec:
			return b.Elem
		}
		return ir.F32
	case ir.Member:
		base := e.infer(x.E)
		if n, ok := base.(ir.Named); ok {
			for _, f := range e.structs[n.Name].Fields {
				if f.Name == x.Field {
					return f.Type
				}
			}
		}
		if v, ok := base.(ir.Vec); ok {
			if len(x.Field) == 1 {
				return v.Elem
			}
			return ir.Vec{N: len(x.Field), Elem: v.Elem}
		}
		return ir.F32
	case ir.AddrOf:
		return e.infer(x.E)
	}
	return ir.F32
}

func isScalar(t ir.Type) bool { _, ok := t.(ir.Scalar); return ok }

// glslReserved are GLSL keywords/reserved words an authored Elio identifier
// must not collide with; safe() appends "_" to escape them.
var glslReserved = map[string]bool{
	"input": true, "output": true, "common": true, "partition": true,
	"active": true, "asm": true, "class": true, "union": true, "enum": true,
	"typedef": true, "template": true, "this": true, "resource": true,
	"goto": true, "inline": true, "noinline": true, "public": true,
	"static": true, "extern": true, "external": true, "interface": true,
	"long": true, "short": true, "half": true, "fixed": true, "unsigned": true,
	"superp": true, "filter": true, "sizeof": true, "cast": true,
	"namespace": true, "using": true,
}

func safe(name string) string {
	if glslReserved[name] {
		return name + "_"
	}
	return name
}

func typeName(t ir.Type) string {
	switch x := t.(type) {
	case ir.Scalar:
		switch x.Name {
		case "f32":
			return "float"
		case "u32":
			return "uint"
		case "i32":
			return "int"
		default:
			return x.Name
		}
	case ir.Vec:
		switch x.Elem.Name {
		case "u32":
			return fmt.Sprintf("uvec%d", x.N)
		case "i32":
			return fmt.Sprintf("ivec%d", x.N)
		default:
			return fmt.Sprintf("vec%d", x.N)
		}
	case ir.Mat:
		if x.Cols == x.Rows {
			return fmt.Sprintf("mat%d", x.Cols)
		}
		return fmt.Sprintf("mat%dx%d", x.Cols, x.Rows)
	case ir.Atomic:
		return typeName(x.Elem)
	case ir.Array:
		if x.Len == 0 {
			return typeName(x.Elem)
		}
		return fmt.Sprintf("%s[%d]", typeName(x.Elem), x.Len)
	case ir.Named:
		return x.Name
	}
	return "/* unknown type */"
}
