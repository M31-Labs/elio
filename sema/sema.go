// Package sema is Elio's semantic-analysis pass: it validates an ir.Module
// before a backend emits it, so authoring mistakes surface as diagnostics rather
// than as broken shader source. It checks
//
//   - name resolution — every referenced Name binds to a local, a kernel
//     builtin, or a module binding;
//   - mutability — assignment targets root at a `var` local or a read_write
//     storage buffer, never a `let`, a builtin input, a uniform, or read-only
//     storage;
//   - address-of — `&x` targets a storage buffer (the only addressable space,
//     as required by arrayLength / atomic ops);
//   - declarations — no duplicate struct, field, binding slot, param, or
//     same-scope local; and every Named type resolves to a declared struct;
//   - call arity — known fixed-arity builtins are called with the right count.
//
// The IR carries no source spans yet, so diagnostics name the symbol and the
// enclosing kernel rather than a line/column.
package sema

import (
	"fmt"
	"sort"
	"strings"

	"m31labs.dev/elio/ir"
)

// Check validates m and returns every diagnostic it finds (empty == valid).
func Check(m *ir.Module) []error {
	c := &checker{structs: map[string]map[string]ir.Type{}}

	// First pass: collect struct names so field/binding types can reference them.
	for i := range m.Structs {
		s := &m.Structs[i]
		if _, dup := c.structs[s.Name]; dup {
			c.errf("duplicate struct %q", s.Name)
		}
		fields := map[string]ir.Type{}
		for _, f := range s.Fields {
			if _, dup := fields[f.Name]; dup {
				c.errf("struct %q: duplicate field %q", s.Name, f.Name)
			}
			fields[f.Name] = f.Type
		}
		c.structs[s.Name] = fields
	}
	// Field types resolve (Named refs point at declared structs).
	for _, s := range m.Structs {
		for _, f := range s.Fields {
			c.checkType(s.Name, f.Type)
		}
	}

	// Module-level constants and bindings populate the global scope. Consts are
	// checked in order, so a const may reference earlier consts but not bindings.
	global := newScope(nil)
	for _, cn := range m.Consts {
		c.checkType("const "+cn.Name, cn.Type)
		c.checkExpr(global, cn.Value)
		if _, dup := global.syms[cn.Name]; dup {
			c.errf("duplicate const %q", cn.Name)
		}
		global.define(cn.Name, kConst, cn.Type)
	}
	slots := map[[2]int]string{}
	for _, b := range m.Bindings {
		slot := [2]int{b.Group, b.Binding}
		if prev, dup := slots[slot]; dup {
			c.errf("binding %q reuses @group(%d) @binding(%d) already held by %q", b.Name, b.Group, b.Binding, prev)
		}
		slots[slot] = b.Name
		if _, dup := global.syms[b.Name]; dup {
			c.errf("duplicate binding name %q", b.Name)
		}
		global.define(b.Name, bindingKind(b), b.Type)
		c.checkType("binding "+b.Name, b.Type)
		c.checkBindingAccess(b)
	}

	for i := range m.Kernels {
		k := &m.Kernels[i]
		c.kernel = k.Name
		ks := newScope(global)
		pnames := map[string]bool{}
		for _, bi := range k.Builtins {
			if pnames[bi.Name] {
				c.errf("kernel %q: duplicate parameter %q", k.Name, bi.Name)
			}
			pnames[bi.Name] = true
			ks.define(bi.Name, kBuiltin, bi.Type)
		}
		for _, sh := range k.Shared {
			if _, dup := ks.syms[sh.Name]; dup {
				c.errf("duplicate shared variable %q", sh.Name)
			}
			c.checkType("shared "+sh.Name, sh.Type)
			ks.define(sh.Name, kShared, sh.Type)
		}
		c.checkBlock(ks, k.Body)
	}
	return c.errs
}

type symKind int

const (
	kLet symKind = iota
	kVar
	kBuiltin
	kUniform
	kStorageRead
	kStorageRW
	kShared         // workgroup-shared variable: mutable, but not a storage buffer (no &)
	kConst          // module-level compile-time constant: immutable
	kStorageTexture // write-only storage texture: only textureStore may write it
	kTexture        // sampled texture: read-only, only textureLoad and textureDimensions
)

func bindingKind(b ir.Binding) symKind {
	if b.Space == ir.Uniform {
		return kUniform
	}
	if _, ok := b.Type.(ir.StorageTexture); ok {
		return kStorageTexture
	}
	if _, ok := b.Type.(ir.Texture2D); ok {
		return kTexture
	}
	if b.Access == ir.ReadWrite {
		return kStorageRW
	}
	return kStorageRead
}

func (k symKind) immutableHint() string {
	switch k {
	case kLet:
		return "a let binding is immutable — declare it with var"
	case kBuiltin:
		return "a builtin input is read-only"
	case kUniform:
		return "a uniform binding is read-only"
	case kStorageRead:
		return "a read-only storage buffer; bind it storage read_write to write"
	case kConst:
		return "a const is immutable"
	case kStorageTexture:
		return "a storage texture is write-only; use textureStore"
	case kTexture:
		return "a sampled texture is read-only; use textureLoad"
	}
	return "read-only"
}

// symInfo is what a name binds to: its mutability kind and, where inferable, its
// type (nil type == unknown, which suppresses type-dependent checks).
type symInfo struct {
	kind symKind
	typ  ir.Type
}

type scope struct {
	parent *scope
	syms   map[string]symInfo
}

func newScope(parent *scope) *scope { return &scope{parent: parent, syms: map[string]symInfo{}} }

func (s *scope) lookup(n string) (symInfo, bool) {
	for sc := s; sc != nil; sc = sc.parent {
		if si, ok := sc.syms[n]; ok {
			return si, true
		}
	}
	return symInfo{}, false
}

func (s *scope) define(n string, k symKind, t ir.Type) { s.syms[n] = symInfo{kind: k, typ: t} }

type checker struct {
	structs map[string]map[string]ir.Type // struct name → field name → field type
	kernel  string
	span    ir.Span // source position of the statement currently being checked
	errs    []error
}

func (c *checker) errf(format string, a ...any) {
	msg := fmt.Sprintf(format, a...)
	if !c.span.IsZero() {
		msg = fmt.Sprintf("%d:%d: %s", c.span.Line, c.span.Col, msg)
	}
	if c.kernel != "" {
		msg = fmt.Sprintf("kernel %q: %s", c.kernel, msg)
	}
	c.errs = append(c.errs, fmt.Errorf("sema: %s", msg))
}

func (c *checker) checkType(where string, t ir.Type) {
	switch t := t.(type) {
	case ir.Named:
		if _, ok := c.structs[t.Name]; !ok {
			c.errf("%s: unknown type %q", where, t.Name)
		}
	case ir.Array:
		c.checkType(where, t.Elem)
	}
}

var storageTextureFormats = map[string]bool{"rgba16float": true, "rgba32float": true, "r32float": true}

func (c *checker) checkBindingAccess(b ir.Binding) {
	tex, isTex := b.Type.(ir.StorageTexture)
	switch {
	case isTex:
		if b.Space != ir.Storage || b.Access != ir.Write {
			c.errf("binding %q: storage texture needs write access (storage write)", b.Name)
		}
		if !storageTextureFormats[tex.Format] {
			c.errf("binding %q: unknown storage texture format %q", b.Name, tex.Format)
		}
	case isSampled(b.Type):
		if b.Space != ir.Storage || b.Access != ir.Read {
			c.errf("binding %q: sampled texture needs read access (texture name: tex2d)", b.Name)
		}
	case b.Access == ir.Write:
		c.errf("binding %q: write access needs a storage texture", b.Name)
	}
}

func (c *checker) checkBlock(parent *scope, stmts []ir.Stmt) {
	s := newScope(parent)
	for _, st := range stmts {
		c.checkStmt(s, st)
	}
}

func (c *checker) checkStmt(s *scope, st ir.Stmt) {
	c.span = ir.StmtSpan(st)
	switch st := st.(type) {
	case ir.Let:
		c.checkExpr(s, st.Value)
		c.declareLocal(s, st.Name, kLet, c.typeOf(s, st.Value))
	case ir.Var:
		c.checkExpr(s, st.Init)
		if st.Type != nil {
			c.checkType("var "+st.Name, st.Type)
		}
		// An explicit annotation wins; otherwise infer from the initializer.
		vt := st.Type
		if vt == nil {
			vt = c.typeOf(s, st.Init)
		}
		c.declareLocal(s, st.Name, kVar, vt)
	case ir.Assign:
		c.checkExpr(s, st.Target)
		c.checkExpr(s, st.Value)
		c.checkAssignable(s, st.Target)
	case ir.If:
		c.checkExpr(s, st.Cond)
		c.checkBlock(s, st.Then)
		if st.Else != nil {
			c.checkBlock(s, st.Else)
		}
	case ir.For:
		fs := newScope(s)
		if st.Init != nil {
			c.checkStmt(fs, st.Init)
		}
		c.span = st.Span // checkStmt(Init) moved the span; restore the for's
		c.checkExpr(fs, st.Cond)
		if st.Post != nil {
			c.checkStmt(fs, st.Post)
		}
		c.checkBlock(fs, st.Body)
	case ir.While:
		c.checkExpr(s, st.Cond)
		c.checkBlock(s, st.Body)
	case ir.Return, ir.Break, ir.Barrier:
		// no-op
	case ir.Do:
		c.checkExpr(s, st.Expr)
	default:
		c.errf("unhandled statement %T", st)
	}
}

func (c *checker) declareLocal(s *scope, name string, k symKind, t ir.Type) {
	if _, dup := s.syms[name]; dup {
		c.errf("%q is already declared in this block", name)
	}
	s.define(name, k, t)
}

func (c *checker) checkExpr(s *scope, e ir.Expr) {
	switch e := e.(type) {
	case ir.Name:
		if _, ok := s.lookup(e.Name); !ok {
			c.errf("undefined name %q", e.Name)
		}
	case ir.Lit:
		// nothing to resolve
	case ir.Binary:
		c.checkExpr(s, e.L)
		c.checkExpr(s, e.R)
	case ir.Unary:
		c.checkExpr(s, e.E)
	case ir.AddrOf:
		c.checkExpr(s, e.E)
		root, ok := rootName(e.E)
		if !ok {
			c.errf("& requires an addressable storage operand")
			return
		}
		if si, found := s.lookup(root); found && si.kind != kStorageRead && si.kind != kStorageRW {
			c.errf("cannot take the address of %q (& requires a storage buffer)", root)
		}
	case ir.Call:
		for _, a := range e.Args {
			c.checkExpr(s, a)
		}
		c.checkCallArity(e)
		c.checkTextureCall(s, e)
	case ir.Index:
		c.checkExpr(s, e.E)
		c.checkExpr(s, e.Idx)
		if ot := c.typeOf(s, e.E); ot != nil {
			if _, ok := ot.(ir.Scalar); ok {
				c.errf("cannot index %s (not an array, vector, or matrix)", typeName(ot))
			}
		}
	case ir.Member:
		c.checkExpr(s, e.E)
		c.checkMember(s, e)
	default:
		c.errf("unhandled expression %T", e)
	}
}

func (c *checker) checkAssignable(s *scope, target ir.Expr) {
	root, ok := rootName(target)
	if !ok {
		c.errf("invalid assignment target")
		return
	}
	si, found := s.lookup(root)
	if !found {
		return // undefined name already reported by checkExpr
	}
	if si.kind != kVar && si.kind != kStorageRW && si.kind != kShared {
		c.errf("cannot assign to %q (%s)", root, si.kind.immutableHint())
	}
}

// rootName peels Index/Member chains to the base Name of an lvalue.
func rootName(e ir.Expr) (string, bool) {
	switch e := e.(type) {
	case ir.Name:
		return e.Name, true
	case ir.Index:
		return rootName(e.E)
	case ir.Member:
		return rootName(e.E)
	}
	return "", false
}

// builtinArity is the fixed argument count of common shader builtins. Functions
// absent here (or with variable arity) are left to the backend.
var builtinArity = map[string]int{
	"dot": 2, "cross": 2, "reflect": 2, "distance": 2, "step": 2, "pow": 2,
	"normalize": 1, "length": 1, "arrayLength": 1, "abs": 1, "floor": 1,
	"ceil": 1, "fract": 1, "sqrt": 1, "sin": 1, "cos": 1, "tan": 1, "exp": 1,
	"log": 1, "exp2": 1, "log2": 1, "sign": 1,
	"clamp": 3, "mix": 3, "smoothstep": 3, "select": 3,
	"asin": 1, "acos": 1, "atan": 1, "atan2": 2,
	"atomicAdd": 2, "atomicSub": 2, "atomicMax": 2, "atomicMin": 2,
	"textureStore": 3, "textureLoad": 3,
	"atomicExchange": 2, "atomicStore": 2, "atomicLoad": 1,
}

func (c *checker) checkCallArity(e ir.Call) {
	want, known := builtinArity[e.Func]
	if known && len(e.Args) != want {
		c.errf("%s expects %d argument(s), got %d", e.Func, want, len(e.Args))
	}
}

// Errors joins diagnostics into a single sorted, newline-separated error (nil if
// there are none) — convenient for CLI reporting.
func Errors(errs []error) error {
	if len(errs) == 0 {
		return nil
	}
	msgs := make([]string, len(errs))
	for i, e := range errs {
		msgs[i] = e.Error()
	}
	sort.Strings(msgs)
	out := msgs[0]
	for _, m := range msgs[1:] {
		out += "\n" + m
	}
	return fmt.Errorf("%s", out)
}

func isSampled(t ir.Type) bool { _, ok := t.(ir.Texture2D); return ok }

// checkTextureCall checks the texture builtins: textureLoad reads a sampled
// texture and textureStore writes a storage texture; textureDimensions takes
// either, with an explicit u32 level only on a sampled texture. A texture
// argument whose type is not known is left alone.
func (c *checker) checkTextureCall(s *scope, e ir.Call) {
	switch e.Func {
	case "textureLoad":
		if len(e.Args) != 3 {
			return
		}
		if t := c.typeOf(s, e.Args[0]); t != nil && !isSampled(t) {
			c.errf("textureLoad needs a sampled texture (texture name: tex2d), got %s", typeName(t))
		}
		c.checkLevel(s, e.Func, e.Args[2])
	case "textureStore":
		if len(e.Args) != 3 {
			return
		}
		if t := c.typeOf(s, e.Args[0]); t != nil {
			if _, ok := t.(ir.StorageTexture); !ok {
				c.errf("textureStore needs a storage texture (storage write name: tex2d_<format>), got %s", typeName(t))
			}
		}
	case "textureDimensions":
		if len(e.Args) < 1 || len(e.Args) > 2 {
			c.errf("textureDimensions expects 1 or 2 argument(s), got %d", len(e.Args))
			return
		}
		if len(e.Args) == 2 {
			if t := c.typeOf(s, e.Args[0]); t != nil && !isSampled(t) {
				c.errf("textureDimensions with a level needs a sampled texture, got %s", typeName(t))
			}
			c.checkLevel(s, e.Func, e.Args[1])
		}
	}
}

// checkLevel requires a texture level argument to be a u32 where that is
// decidable: a literal must carry the u suffix, a typed name must be u32.
// Arithmetic and calls whose type sema does not infer are accepted.
func (c *checker) checkLevel(s *scope, fn string, level ir.Expr) {
	switch l := level.(type) {
	case ir.Lit:
		if !strings.HasSuffix(l.Text, "u") {
			c.errf("%s: level must be u32, got literal %s", fn, l.Text)
		}
		return
	}
	if t := c.typeOf(s, level); t != nil && t != ir.Type(ir.U32) {
		c.errf("%s: level must be u32, got %s", fn, typeName(t))
	}
}
