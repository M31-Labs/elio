// Package gather lowers a gather-compatible Elio compute kernel to a GLSL ES
// 3.00 fragment shader, so it can run on WebGL2 (which has no compute) as one
// fullscreen render pass over float textures.
//
// Each storage buffer becomes a 2D texture of width `elio_width`: element i is
// texel (i % elio_width, i / elio_width). A read buffer is a sampler2D read
// with texelFetch. A written buffer, or a storage texture written as
// textureStore(t, vec2u(i % W, i / W), v), is a color attachment (RGBA32F or
// RGBA16F; at most 32 bytes per pixel across all outputs).
// Invocation gid.x is the pixel's linear index, so every invocation writes its
// own element, which is the gather form. sema.CheckGather defines which
// kernels qualify. The Layout result tells the host how to bind them.
package gather

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"m31labs.dev/elio/emit/internal/prismtypes"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/sema"
	"m31labs.dev/prism/dialect"
)

var glslDialect = dialect.GLSL{} // type spellings match ES 3.00

// fragIndex is gid.x: the pixel's linear index.
const fragIndex = "(uint(gl_FragCoord.y) * elio_width + uint(gl_FragCoord.x))"

// Emit renders kernel as a #version 300 es fragment shader and returns the
// host Layout. It runs sema.Check and sema.CheckGather first and returns every
// diagnostic if either fails.
func Emit(m *ir.Module, kernel string) (string, Layout, error) {
	errs := sema.Check(m)
	errs = append(errs, sema.CheckGather(m, kernel)...)
	if len(errs) > 0 {
		return "", Layout{}, errors.Join(errs...)
	}
	var k *ir.Kernel
	for i := range m.Kernels {
		if m.Kernels[i].Name == kernel {
			k = &m.Kernels[i]
		}
	}
	e := &emitter{
		m:        m,
		k:        k,
		structs:  map[string]ir.Struct{},
		bindings: map[string]ir.Binding{},
		locals:   map[string]ir.Type{},
		builtins: map[string]ir.Type{},
		outVar:   map[string]string{},
		reads:    map[string]bool{},
	}
	for _, s := range m.Structs {
		e.structs[s.Name] = s
	}
	for _, b := range m.Bindings {
		e.bindings[b.Name] = b
	}
	for _, bi := range k.Builtins {
		e.builtins[bi.Name] = bi.Type
		e.gid = bi.Name
	}
	return e.emit()
}

type emitter struct {
	m        *ir.Module
	k        *ir.Kernel
	structs  map[string]ir.Struct
	bindings map[string]ir.Binding
	locals   map[string]ir.Type
	builtins map[string]ir.Type
	gid      string
	outVar   map[string]string // written buffer → out variable
	reads    map[string]bool   // read buffers
	b        strings.Builder
	err      error
}

func (e *emitter) emit() (string, Layout, error) {
	lay := Layout{Kernel: e.k.Name}
	written := map[string]bool{}
	used := map[string]bool{}
	walkKernel(e.k.Body, func(s ir.Stmt) {
		if a, ok := s.(ir.Assign); ok {
			if root := rootOf(a.Target); e.isStorage(root) {
				written[root] = true
				used[root] = true
			}
		}
		if tex, _, ok := e.textureStore(s); ok {
			written[tex] = true
			used[tex] = true
		}
	}, func(x ir.Expr) {
		if n, ok := x.(ir.Name); ok {
			if _, isB := e.bindings[n.Name]; isB {
				used[n.Name] = true
				if e.isStorage(n.Name) {
					e.reads[n.Name] = true
				}
			}
		}
	})
	// A written buffer never appears as a read (rule 4), so reads is disjoint.

	var ordered []ir.Binding
	for _, b := range e.m.Bindings {
		if used[b.Name] {
			ordered = append(ordered, b)
		}
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Group != ordered[j].Group {
			return ordered[i].Group < ordered[j].Group
		}
		return ordered[i].Binding < ordered[j].Binding
	})

	w := &e.b
	w.WriteString("#version 300 es\n")
	w.WriteString("precision highp float;\nprecision highp int;\nprecision highp sampler2D;\n\n")
	w.WriteString("uniform uint elio_width;\n\n")
	for _, cn := range e.m.Consts {
		fmt.Fprintf(w, "const %s %s = %s;\n", typeName(cn.Type), safe(cn.Name), e.expr(cn.Value))
	}
	if len(e.m.Consts) > 0 {
		w.WriteString("\n")
	}

	loc := 0
	for _, b := range ordered {
		switch {
		case b.Space == ir.Uniform:
			n := b.Type.(ir.Named)
			block := n.Name + "_ubo"
			fmt.Fprintf(w, "layout(std140) uniform %s {\n", block)
			for _, f := range e.structs[n.Name].Fields {
				fmt.Fprintf(w, "  %s %s;\n", typeName(f.Type), safe(f.Name))
			}
			fmt.Fprintf(w, "} %s;\n", safe(b.Name))
			lay.Uniforms = append(lay.Uniforms, Uniform{Name: b.Name, Block: block, Struct: n.Name, Binding: b.Binding})
		case written[b.Name]:
			v := b.Name + "_out"
			e.outVar[b.Name] = v
			fmt.Fprintf(w, "layout(location = %d) out vec4 %s;\n", loc, v)
			o := Output{Name: b.Name, Elem: elemKind(b), Location: loc, Variable: v}
			if tex, ok := b.Type.(ir.StorageTexture); ok {
				o.Format = tex.Format
			}
			lay.Outputs = append(lay.Outputs, o)
			loc++
		default:
			fmt.Fprintf(w, "uniform highp sampler2D %s_tex;\nuniform uint %s_len;\n", b.Name, b.Name)
			lay.Inputs = append(lay.Inputs, Buffer{
				Name: b.Name, Elem: elemKind(b), Sampler: b.Name + "_tex", Length: b.Name + "_len", Binding: b.Binding,
			})
		}
	}
	w.WriteString("\n")
	for _, in := range lay.Inputs {
		ty, sw := "float", ".x"
		switch in.Elem {
		case "vec2":
			ty, sw = "vec2", ".xy"
		case "vec4":
			ty, sw = "vec4", ""
		}
		fmt.Fprintf(w, "%s %s_at(uint i) {\n  return texelFetch(%s, ivec2(int(i %% elio_width), int(i / elio_width)), 0)%s;\n}\n\n",
			ty, in.Name, in.Sampler, sw)
	}

	w.WriteString("void main() {\n")
	for _, o := range lay.Outputs {
		fmt.Fprintf(w, "  %s = vec4(0.0);\n", o.Variable)
	}
	for _, s := range e.k.Body {
		e.stmt(s, 1)
	}
	w.WriteString("}\n")
	if e.err != nil {
		return "", Layout{}, e.err
	}
	return w.String(), lay, nil
}

func elemKind(b ir.Binding) string {
	if _, ok := b.Type.(ir.StorageTexture); ok {
		return "vec4"
	}
	if arr, ok := b.Type.(ir.Array); ok {
		if v, ok := arr.Elem.(ir.Vec); ok {
			return fmt.Sprintf("vec%d", v.N)
		}
	}
	return "f32"
}

func (e *emitter) isStorage(name string) bool {
	b, ok := e.bindings[name]
	if ok {
		if _, tex := b.Type.(ir.StorageTexture); tex {
			return false
		}
	}
	return ok && b.Space == ir.Storage
}

// textureStore matches `textureStore(tex, coord, value);` on a storage texture
// binding. sema.CheckGather has already required the coordinate to be the
// fragment's own pixel, so the emitter drops it.
func (e *emitter) textureStore(s ir.Stmt) (tex string, value ir.Expr, ok bool) {
	d, isDo := s.(ir.Do)
	if !isDo {
		return "", nil, false
	}
	c, isCall := d.Expr.(ir.Call)
	if !isCall || c.Func != "textureStore" || len(c.Args) != 3 {
		return "", nil, false
	}
	n, isName := c.Args[0].(ir.Name)
	if !isName {
		return "", nil, false
	}
	if b, found := e.bindings[n.Name]; !found {
		return "", nil, false
	} else if _, isTex := b.Type.(ir.StorageTexture); !isTex {
		return "", nil, false
	}
	return n.Name, c.Args[2], true
}

func rootOf(x ir.Expr) string {
	for {
		switch v := x.(type) {
		case ir.Index:
			x = v.E
		case ir.Member:
			x = v.E
		case ir.Name:
			return v.Name
		default:
			return ""
		}
	}
}

func (e *emitter) fail(format string, a ...any) {
	if e.err == nil {
		e.err = fmt.Errorf("gather: kernel %q: %s", e.k.Name, fmt.Sprintf(format, a...))
	}
}

func (e *emitter) stmt(s ir.Stmt, depth int) {
	pad := strings.Repeat("  ", depth)
	w := &e.b
	switch x := s.(type) {
	case ir.Let:
		t := e.infer(x.Value)
		e.locals[x.Name] = t
		fmt.Fprintf(w, "%s%s %s = %s;\n", pad, typeName(t), safe(x.Name), e.expr(x.Value))
	case ir.Var:
		t := x.Type
		if t == nil {
			t = e.infer(x.Init)
		}
		e.locals[x.Name] = t
		fmt.Fprintf(w, "%s%s %s = %s;\n", pad, typeName(t), safe(x.Name), e.expr(x.Init))
	case ir.Assign:
		fmt.Fprintf(w, "%s%s;\n", pad, e.assign(x))
	case ir.Return:
		fmt.Fprintf(w, "%sreturn;\n", pad)
	case ir.Break:
		fmt.Fprintf(w, "%sbreak;\n", pad)
	case ir.Do:
		if tex, val, ok := e.textureStore(x); ok {
			fmt.Fprintf(w, "%s%s = %s;\n", pad, e.outVar[tex], e.expr(val))
			break
		}
		fmt.Fprintf(w, "%s%s;\n", pad, e.expr(x.Expr))
	case ir.If:
		fmt.Fprintf(w, "%sif (%s) {\n", pad, e.expr(x.Cond))
		e.block(x.Then, depth+1)
		if len(x.Else) > 0 {
			fmt.Fprintf(w, "%s} else {\n", pad)
			e.block(x.Else, depth+1)
		}
		fmt.Fprintf(w, "%s}\n", pad)
	case ir.For:
		fmt.Fprintf(w, "%sfor (%s; %s; %s) {\n", pad, e.inline(x.Init), e.expr(x.Cond), e.inline(x.Post))
		e.block(x.Body, depth+1)
		fmt.Fprintf(w, "%s}\n", pad)
	case ir.While:
		fmt.Fprintf(w, "%swhile (%s) {\n", pad, e.expr(x.Cond))
		e.block(x.Body, depth+1)
		fmt.Fprintf(w, "%s}\n", pad)
	default:
		e.fail("unsupported statement %T", s)
	}
}

func (e *emitter) block(body []ir.Stmt, depth int) {
	for _, s := range body {
		e.stmt(s, depth)
	}
}

func (e *emitter) inline(s ir.Stmt) string {
	switch x := s.(type) {
	case ir.Var:
		t := x.Type
		if t == nil {
			t = e.infer(x.Init)
		}
		e.locals[x.Name] = t
		return fmt.Sprintf("%s %s = %s", typeName(t), safe(x.Name), e.expr(x.Init))
	case ir.Let:
		t := e.infer(x.Value)
		e.locals[x.Name] = t
		return fmt.Sprintf("%s %s = %s", typeName(t), safe(x.Name), e.expr(x.Value))
	case ir.Assign:
		return e.assign(x)
	}
	return ""
}

// assign spells an assignment without the trailing semicolon. A write to a
// written buffer becomes an assignment to its out variable, padded to vec4.
func (e *emitter) assign(a ir.Assign) string {
	if ix, ok := a.Target.(ir.Index); ok {
		if n, ok := ix.E.(ir.Name); ok {
			if v, isOut := e.outVar[n.Name]; isOut {
				val := e.expr(a.Value)
				switch t := e.infer(a.Value).(type) {
				case ir.Vec:
					switch t.N {
					case 4:
					case 3:
						val = fmt.Sprintf("vec4(%s, 0.0)", val)
					case 2:
						val = fmt.Sprintf("vec4(%s, 0.0, 0.0)", val)
					default:
						val = fmt.Sprintf("vec4(%s, 0.0, 0.0, 0.0)", val)
					}
				default:
					val = fmt.Sprintf("vec4(%s, 0.0, 0.0, 0.0)", val)
				}
				return fmt.Sprintf("%s = %s", v, val)
			}
		}
	}
	return fmt.Sprintf("%s %s= %s", e.expr(a.Target), a.Op, e.expr(a.Value))
}

func (e *emitter) expr(ex ir.Expr) string {
	switch x := ex.(type) {
	case ir.Name:
		if x.Name == e.gid && e.gid != "" {
			return fmt.Sprintf("uvec3(%s, 0u, 0u)", fragIndex)
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
		if n, ok := x.E.(ir.Name); ok && e.reads[n.Name] {
			return fmt.Sprintf("%s_at(%s)", n.Name, e.expr(x.Idx))
		}
		return fmt.Sprintf("%s[%s]", e.expr(x.E), e.expr(x.Idx))
	case ir.Member:
		if n, ok := x.E.(ir.Name); ok && n.Name == e.gid && e.gid != "" {
			return gidSwizzle(x.Field)
		}
		if c, ok := x.E.(ir.Call); ok && c.Func == "textureDimensions" && x.Field == "x" {
			return "elio_width" // the render target is elio_width texels wide
		}
		return fmt.Sprintf("%s.%s", e.expr(x.E), safe(x.Field))
	case ir.AddrOf:
		return e.expr(x.E)
	}
	e.fail("unsupported expression %T", ex)
	return ""
}

// gidSwizzle spells gid.<field>: x is the pixel index, y and z are 0.
func gidSwizzle(field string) string {
	comps := make([]string, len(field))
	for i, c := range field {
		if c == 'x' {
			comps[i] = fragIndex
		} else {
			comps[i] = "0u"
		}
	}
	if len(comps) == 1 {
		return comps[0]
	}
	return fmt.Sprintf("uvec%d(%s)", len(comps), strings.Join(comps, ", "))
}

func (e *emitter) call(c ir.Call) string {
	switch c.Func {
	case "arrayLength":
		if len(c.Args) == 1 {
			if a, ok := c.Args[0].(ir.AddrOf); ok {
				if n, ok := a.E.(ir.Name); ok {
					return n.Name + "_len"
				}
			}
		}
		e.fail("arrayLength needs &buffer")
		return ""
	case "select":
		if len(c.Args) == 3 {
			return fmt.Sprintf("(%s ? %s : %s)", e.expr(c.Args[2]), e.expr(c.Args[1]), e.expr(c.Args[0]))
		}
	}
	fn := c.Func
	if n, elem, ok := ir.VecConstructor(c.Func); ok {
		fn = typeName(ir.Vec{N: n, Elem: elem})
	} else if elem, ok := ir.ScalarCast(c.Func); ok {
		fn = typeName(elem)
	}
	args := make([]string, len(c.Args))
	for i, a := range c.Args {
		args[i] = e.expr(a)
	}
	return fmt.Sprintf("%s(%s)", fn, strings.Join(args, ", "))
}

func (e *emitter) infer(ex ir.Expr) ir.Type {
	switch x := ex.(type) {
	case ir.Name:
		if t, ok := e.locals[x.Name]; ok {
			return t
		}
		if t, ok := e.builtins[x.Name]; ok {
			return t
		}
		if b, ok := e.bindings[x.Name]; ok {
			return b.Type
		}
		return ir.F32
	case ir.Lit:
		switch {
		case x.Text == "true" || x.Text == "false":
			return ir.Bool
		case strings.HasSuffix(x.Text, "u"):
			return ir.U32
		case strings.ContainsAny(x.Text, ".eE"):
			return ir.F32
		default:
			return ir.I32
		}
	case ir.Binary:
		switch x.Op {
		case "<", ">", "<=", ">=", "==", "!=", "&&", "||":
			return ir.Bool
		}
		lt := e.infer(x.L)
		if _, ok := lt.(ir.Scalar); ok {
			if rt := e.infer(x.R); !isScalar(rt) {
				return rt
			}
		}
		return lt
	case ir.Unary:
		if x.Op == "!" {
			return ir.Bool
		}
		return e.infer(x.E)
	case ir.Call:
		if n, elem, ok := ir.VecConstructor(x.Func); ok {
			return ir.Vec{N: n, Elem: elem}
		}
		if elem, ok := ir.ScalarCast(x.Func); ok {
			return elem
		}
		switch x.Func {
		case "arrayLength":
			return ir.U32
		case "dot", "length", "distance":
			return ir.F32
		case "select":
			if len(x.Args) >= 2 {
				return e.infer(x.Args[0])
			}
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
		if c, ok := x.E.(ir.Call); ok && c.Func == "textureDimensions" && x.Field == "x" {
			return ir.U32
		}
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

// walkKernel visits every statement and every expression of body.
func walkKernel(body []ir.Stmt, onStmt func(ir.Stmt), onExpr func(ir.Expr)) {
	var ex func(ir.Expr)
	ex = func(x ir.Expr) {
		if x == nil {
			return
		}
		onExpr(x)
		switch v := x.(type) {
		case ir.Binary:
			ex(v.L)
			ex(v.R)
		case ir.Unary:
			ex(v.E)
		case ir.Call:
			for _, a := range v.Args {
				ex(a)
			}
		case ir.Index:
			ex(v.E)
			ex(v.Idx)
		case ir.Member:
			ex(v.E)
		case ir.AddrOf:
			ex(v.E)
		}
	}
	var st func(ir.Stmt)
	var blk func([]ir.Stmt)
	blk = func(b []ir.Stmt) {
		for _, s := range b {
			st(s)
		}
	}
	st = func(s ir.Stmt) {
		if s == nil {
			return
		}
		onStmt(s)
		switch v := s.(type) {
		case ir.Let:
			ex(v.Value)
		case ir.Var:
			ex(v.Init)
		case ir.Assign:
			// A write target's base buffer is not a read; its indices are.
			t := v.Target
			for {
				if i, ok := t.(ir.Index); ok {
					ex(i.Idx)
					t = i.E
				} else if m, ok := t.(ir.Member); ok {
					t = m.E
				} else {
					break
				}
			}
			ex(v.Value)
		case ir.If:
			ex(v.Cond)
			blk(v.Then)
			blk(v.Else)
		case ir.For:
			st(v.Init)
			ex(v.Cond)
			st(v.Post)
			blk(v.Body)
		case ir.While:
			ex(v.Cond)
			blk(v.Body)
		case ir.Do:
			ex(v.Expr)
		}
	}
	blk(body)
}

var glslReserved = map[string]bool{
	"input": true, "output": true, "common": true, "partition": true,
	"active": true, "asm": true, "class": true, "union": true, "enum": true,
	"typedef": true, "template": true, "this": true, "resource": true,
	"goto": true, "inline": true, "noinline": true, "public": true,
	"static": true, "extern": true, "external": true, "interface": true,
	"long": true, "short": true, "half": true, "fixed": true, "unsigned": true,
	"superp": true, "filter": true, "sizeof": true, "cast": true,
	"namespace": true, "using": true,
	"in": true, "out": true, "inout": true, "uniform": true, "buffer": true,
	"shared": true, "attribute": true, "varying": true, "layout": true,
	"coherent": true, "volatile": true, "restrict": true, "readonly": true,
	"writeonly": true, "flat": true, "smooth": true, "centroid": true,
	"sample": true, "patch": true, "subroutine": true, "precision": true,
	"discard": true, "invariant": true, "lowp": true, "mediump": true, "highp": true,
}

func safe(name string) string {
	if glslReserved[name] {
		return name + "_"
	}
	return name
}

func typeName(t ir.Type) string {
	if gt, ok := prismtypes.FromIR(t); ok {
		return glslDialect.TypeName(gt)
	}
	if n, ok := t.(ir.Named); ok {
		return n.Name
	}
	return "/* unknown type */"
}
