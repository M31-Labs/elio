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

	"m31labs.dev/elio/ir"
)

// Check validates m and returns every diagnostic it finds (empty == valid).
func Check(m *ir.Module) []error {
	c := &checker{structs: map[string]bool{}}

	// First pass: collect struct names so field/binding types can reference them.
	for _, s := range m.Structs {
		if c.structs[s.Name] {
			c.errf("duplicate struct %q", s.Name)
		}
		c.structs[s.Name] = true
	}
	// Struct fields: no dups, and Named field types resolve.
	for _, s := range m.Structs {
		seen := map[string]bool{}
		for _, f := range s.Fields {
			if seen[f.Name] {
				c.errf("struct %q: duplicate field %q", s.Name, f.Name)
			}
			seen[f.Name] = true
			c.checkType(s.Name, f.Type)
		}
	}

	// Module bindings populate the global scope.
	global := newScope(nil)
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
		global.define(b.Name, bindingKind(b))
		c.checkType("binding "+b.Name, b.Type)
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
			ks.define(bi.Name, kBuiltin)
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
)

func bindingKind(b ir.Binding) symKind {
	if b.Space == ir.Uniform {
		return kUniform
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
	}
	return "read-only"
}

type scope struct {
	parent *scope
	syms   map[string]symKind
}

func newScope(parent *scope) *scope { return &scope{parent: parent, syms: map[string]symKind{}} }

func (s *scope) lookup(n string) (symKind, bool) {
	for sc := s; sc != nil; sc = sc.parent {
		if k, ok := sc.syms[n]; ok {
			return k, true
		}
	}
	return 0, false
}

func (s *scope) define(n string, k symKind) { s.syms[n] = k }

type checker struct {
	structs map[string]bool
	kernel  string
	errs    []error
}

func (c *checker) errf(format string, a ...any) {
	msg := fmt.Sprintf(format, a...)
	if c.kernel != "" {
		msg = fmt.Sprintf("kernel %q: %s", c.kernel, msg)
	}
	c.errs = append(c.errs, fmt.Errorf("sema: %s", msg))
}

func (c *checker) checkType(where string, t ir.Type) {
	switch t := t.(type) {
	case ir.Named:
		if !c.structs[t.Name] {
			c.errf("%s: unknown type %q", where, t.Name)
		}
	case ir.Array:
		c.checkType(where, t.Elem)
	}
}

func (c *checker) checkBlock(parent *scope, stmts []ir.Stmt) {
	s := newScope(parent)
	for _, st := range stmts {
		c.checkStmt(s, st)
	}
}

func (c *checker) checkStmt(s *scope, st ir.Stmt) {
	switch st := st.(type) {
	case ir.Let:
		c.checkExpr(s, st.Value)
		c.declareLocal(s, st.Name, kLet)
	case ir.Var:
		c.checkExpr(s, st.Init)
		if st.Type != nil {
			c.checkType("var "+st.Name, st.Type)
		}
		c.declareLocal(s, st.Name, kVar)
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
		c.checkExpr(fs, st.Cond)
		if st.Post != nil {
			c.checkStmt(fs, st.Post)
		}
		c.checkBlock(fs, st.Body)
	case ir.Return, ir.Break:
		// no-op
	case ir.Do:
		c.checkExpr(s, st.Expr)
	default:
		c.errf("unhandled statement %T", st)
	}
}

func (c *checker) declareLocal(s *scope, name string, k symKind) {
	if _, dup := s.syms[name]; dup {
		c.errf("%q is already declared in this block", name)
	}
	s.define(name, k)
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
		if k, found := s.lookup(root); found && k != kStorageRead && k != kStorageRW {
			c.errf("cannot take the address of %q (& requires a storage buffer)", root)
		}
	case ir.Call:
		for _, a := range e.Args {
			c.checkExpr(s, a)
		}
		c.checkCallArity(e)
	case ir.Index:
		c.checkExpr(s, e.E)
		c.checkExpr(s, e.Idx)
	case ir.Member:
		c.checkExpr(s, e.E)
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
	k, found := s.lookup(root)
	if !found {
		return // undefined name already reported by checkExpr
	}
	if k != kVar && k != kStorageRW {
		c.errf("cannot assign to %q (%s)", root, k.immutableHint())
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
	"clamp": 3, "mix": 3, "smoothstep": 3,
	"atomicAdd": 2, "atomicSub": 2, "atomicMax": 2, "atomicMin": 2,
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
