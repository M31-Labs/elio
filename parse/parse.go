// Package parse is Elio's front-end: it lexes and parses .elio compute source
// into an ir.Module. (A grammargen/gotreesitter grammar — the house style
// shared with Selena and Manta — is the intended production parser; this
// hand-written recursive-descent parser bootstraps the language so kernels can
// be authored, not hand-built, and is validated by emitting byte-identical WGSL
// to the hand-built IR.)
//
// Surface (Go-flavored types avoid the <…> ambiguity of WGSL generics):
//
//	struct Name { field: type; … }
//	@group(g) @binding(b) uniform name: T;
//	@group(g) @binding(b) storage read|read_write name: T;
//	@workgroup(x[,y,z]) kernel name(p: builtin, …) { stmts }
//
// Types: f32 i32 u32 bool · vec2/3/4[u] · mat3 mat4 · atomic_u32/atomic_i32 ·
// [N]T (fixed array) · []T (runtime array) · NamedStruct.
package parse

import (
	"fmt"
	"strconv"
	"strings"

	"m31labs.dev/elio/ir"
)

// Parse parses Elio source into an ir.Module.
func Parse(src string) (*ir.Module, error) {
	toks, err := lex(src)
	if err != nil {
		return nil, err
	}
	return (&parser{toks: toks}).module()
}

// --- lexer ---

type tokKind int

const (
	tEOF tokKind = iota
	tIdent
	tNumber
	tPunct
)

type token struct {
	kind tokKind
	val  string
	line int
}

func lex(src string) ([]token, error) {
	var toks []token
	line := 1
	emit := func(k tokKind, v string) { toks = append(toks, token{kind: k, val: v, line: line}) }
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == '\n':
			line++
			i++
		case c == ' ' || c == '\t' || c == '\r':
			i++
		case c == '/' && i+1 < len(src) && src[i+1] == '/':
			for i < len(src) && src[i] != '\n' {
				i++
			}
		case isIdentStart(c):
			j := i
			for j < len(src) && isIdentPart(src[j]) {
				j++
			}
			emit(tIdent, src[i:j])
			i = j
		case c >= '0' && c <= '9':
			j := i
			for j < len(src) && (src[j] >= '0' && src[j] <= '9' || src[j] == '.') {
				j++
			}
			if j < len(src) && (src[j] == 'u' || src[j] == 'i' || src[j] == 'f') {
				j++
			}
			emit(tNumber, src[i:j])
			i = j
		default:
			if i+1 < len(src) {
				switch src[i : i+2] {
				case "<=", ">=", "==", "!=", "+=", "-=", "*=", "/=", "%=":
					emit(tPunct, src[i:i+2])
					i += 2
					continue
				}
			}
			if strings.IndexByte("{}()[]<>:;,.@&=+-*/%!", c) >= 0 {
				emit(tPunct, string(c))
				i++
			} else {
				return nil, fmt.Errorf("parse: line %d: unexpected character %q", line, c)
			}
		}
	}
	emit(tEOF, "")
	return toks, nil
}

func isIdentStart(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}
func isIdentPart(c byte) bool { return isIdentStart(c) || c >= '0' && c <= '9' }

// --- parser ---

type parser struct {
	toks []token
	i    int
}

func (p *parser) peek() token { return p.toks[p.i] }
func (p *parser) next() token {
	t := p.toks[p.i]
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}
func (p *parser) atEOF() bool           { return p.peek().kind == tEOF }
func (p *parser) isPunct(v string) bool { t := p.peek(); return t.kind == tPunct && t.val == v }
func (p *parser) isIdent(v string) bool { t := p.peek(); return t.kind == tIdent && t.val == v }

func (p *parser) errf(format string, a ...any) error {
	return fmt.Errorf("parse: line %d: %s", p.peek().line, fmt.Sprintf(format, a...))
}
func (p *parser) wantPunct(v string) error {
	if !p.isPunct(v) {
		return p.errf("expected %q, got %q", v, p.peek().val)
	}
	p.next()
	return nil
}
func (p *parser) wantIdent() (string, error) {
	t := p.peek()
	if t.kind != tIdent {
		return "", p.errf("expected identifier, got %q", t.val)
	}
	p.next()
	return t.val, nil
}

func (p *parser) module() (*ir.Module, error) {
	m := &ir.Module{}
	for !p.atEOF() {
		switch {
		case p.isIdent("struct"):
			s, err := p.structDecl()
			if err != nil {
				return nil, err
			}
			m.Structs = append(m.Structs, s)
		case p.isPunct("@"):
			next := p.toks[p.i+1]
			if next.kind == tIdent && next.val == "group" {
				b, err := p.binding()
				if err != nil {
					return nil, err
				}
				m.Bindings = append(m.Bindings, b)
			} else if next.kind == tIdent && next.val == "workgroup" {
				k, err := p.kernel()
				if err != nil {
					return nil, err
				}
				m.Kernels = append(m.Kernels, k)
			} else {
				return nil, p.errf("expected @group or @workgroup, got @%s", next.val)
			}
		default:
			return nil, p.errf("unexpected %q at top level", p.peek().val)
		}
	}
	return m, nil
}

func (p *parser) structDecl() (ir.Struct, error) {
	p.next() // struct
	name, err := p.wantIdent()
	if err != nil {
		return ir.Struct{}, err
	}
	if err := p.wantPunct("{"); err != nil {
		return ir.Struct{}, err
	}
	var fields []ir.Field
	for !p.isPunct("}") {
		fname, err := p.wantIdent()
		if err != nil {
			return ir.Struct{}, err
		}
		if err := p.wantPunct(":"); err != nil {
			return ir.Struct{}, err
		}
		ft, err := p.typ()
		if err != nil {
			return ir.Struct{}, err
		}
		if err := p.wantPunct(";"); err != nil {
			return ir.Struct{}, err
		}
		fields = append(fields, ir.Field{Name: fname, Type: ft})
	}
	p.next() // }
	return ir.Struct{Name: name, Fields: fields}, nil
}

func (p *parser) typ() (ir.Type, error) {
	if p.isPunct("[") {
		p.next()
		n := 0
		if p.peek().kind == tNumber {
			n, _ = strconv.Atoi(p.next().val)
		}
		if err := p.wantPunct("]"); err != nil {
			return nil, err
		}
		elem, err := p.typ()
		if err != nil {
			return nil, err
		}
		return ir.Array{Elem: elem, Len: n}, nil
	}
	name, err := p.wantIdent()
	if err != nil {
		return nil, err
	}
	return scalarType(name), nil
}

func scalarType(name string) ir.Type {
	switch name {
	case "f32":
		return ir.F32
	case "i32":
		return ir.I32
	case "u32":
		return ir.U32
	case "bool":
		return ir.Bool
	case "vec2":
		return ir.Vec{N: 2, Elem: ir.F32}
	case "vec3":
		return ir.Vec{N: 3, Elem: ir.F32}
	case "vec4":
		return ir.Vec{N: 4, Elem: ir.F32}
	case "vec2u":
		return ir.Vec{N: 2, Elem: ir.U32}
	case "vec3u":
		return ir.Vec{N: 3, Elem: ir.U32}
	case "vec4u":
		return ir.Vec{N: 4, Elem: ir.U32}
	case "mat3":
		return ir.Mat{Cols: 3, Rows: 3, Elem: ir.F32}
	case "mat4":
		return ir.Mat{Cols: 4, Rows: 4, Elem: ir.F32}
	case "atomic_u32":
		return ir.Atomic{Elem: ir.U32}
	case "atomic_i32":
		return ir.Atomic{Elem: ir.I32}
	}
	return ir.Named{Name: name}
}

func (p *parser) attrInt(name string) (int, error) {
	if err := p.wantPunct("@"); err != nil {
		return 0, err
	}
	if !p.isIdent(name) {
		return 0, p.errf("expected @%s", name)
	}
	p.next()
	if err := p.wantPunct("("); err != nil {
		return 0, err
	}
	if p.peek().kind != tNumber {
		return 0, p.errf("expected number in @%s(…)", name)
	}
	v, _ := strconv.Atoi(p.next().val)
	if err := p.wantPunct(")"); err != nil {
		return 0, err
	}
	return v, nil
}

func (p *parser) binding() (ir.Binding, error) {
	g, err := p.attrInt("group")
	if err != nil {
		return ir.Binding{}, err
	}
	bn, err := p.attrInt("binding")
	if err != nil {
		return ir.Binding{}, err
	}
	b := ir.Binding{Group: g, Binding: bn}
	space, err := p.wantIdent()
	if err != nil {
		return ir.Binding{}, err
	}
	switch space {
	case "uniform":
		b.Space = ir.Uniform
	case "storage":
		b.Space = ir.Storage
		acc, err := p.wantIdent()
		if err != nil {
			return ir.Binding{}, err
		}
		switch acc {
		case "read":
			b.Access = ir.Read
		case "read_write":
			b.Access = ir.ReadWrite
		default:
			return ir.Binding{}, p.errf("expected read|read_write, got %q", acc)
		}
	default:
		return ir.Binding{}, p.errf("expected uniform|storage, got %q", space)
	}
	if b.Name, err = p.wantIdent(); err != nil {
		return ir.Binding{}, err
	}
	if err := p.wantPunct(":"); err != nil {
		return ir.Binding{}, err
	}
	if b.Type, err = p.typ(); err != nil {
		return ir.Binding{}, err
	}
	if err := p.wantPunct(";"); err != nil {
		return ir.Binding{}, err
	}
	return b, nil
}

func (p *parser) kernel() (ir.Kernel, error) {
	if err := p.wantPunct("@"); err != nil {
		return ir.Kernel{}, err
	}
	if !p.isIdent("workgroup") {
		return ir.Kernel{}, p.errf("expected @workgroup")
	}
	p.next()
	if err := p.wantPunct("("); err != nil {
		return ir.Kernel{}, err
	}
	ws := [3]int{1, 1, 1}
	for idx := 0; ; idx++ {
		if p.peek().kind != tNumber {
			return ir.Kernel{}, p.errf("expected workgroup size")
		}
		v, _ := strconv.Atoi(p.next().val)
		if idx < 3 {
			ws[idx] = v
		}
		if p.isPunct(",") {
			p.next()
			continue
		}
		break
	}
	if err := p.wantPunct(")"); err != nil {
		return ir.Kernel{}, err
	}
	if !p.isIdent("kernel") {
		return ir.Kernel{}, p.errf("expected 'kernel'")
	}
	p.next()
	name, err := p.wantIdent()
	if err != nil {
		return ir.Kernel{}, err
	}
	if err := p.wantPunct("("); err != nil {
		return ir.Kernel{}, err
	}
	var builtins []ir.Builtin
	for !p.isPunct(")") {
		pname, err := p.wantIdent()
		if err != nil {
			return ir.Kernel{}, err
		}
		if err := p.wantPunct(":"); err != nil {
			return ir.Kernel{}, err
		}
		bt, err := p.wantIdent()
		if err != nil {
			return ir.Kernel{}, err
		}
		builtins = append(builtins, ir.Builtin{Name: pname, Builtin: bt, Type: ir.Vec{N: 3, Elem: ir.U32}})
		if p.isPunct(",") {
			p.next()
		}
	}
	p.next() // )
	shared, body, err := p.kernelBody()
	if err != nil {
		return ir.Kernel{}, err
	}
	return ir.Kernel{Name: name, WorkgroupSize: ws, Builtins: builtins, Shared: shared, Body: body}, nil
}

// kernelBody parses { shared-decls… statements… }: workgroup-shared
// declarations are a prefix, then the statement body.
func (p *parser) kernelBody() ([]ir.Shared, []ir.Stmt, error) {
	if err := p.wantPunct("{"); err != nil {
		return nil, nil, err
	}
	var shared []ir.Shared
	for p.isIdent("shared") {
		p.next() // shared
		name, err := p.wantIdent()
		if err != nil {
			return nil, nil, err
		}
		if err := p.wantPunct(":"); err != nil {
			return nil, nil, err
		}
		t, err := p.typ()
		if err != nil {
			return nil, nil, err
		}
		if err := p.wantPunct(";"); err != nil {
			return nil, nil, err
		}
		shared = append(shared, ir.Shared{Name: name, Type: t})
	}
	var body []ir.Stmt
	for !p.isPunct("}") {
		s, err := p.stmt()
		if err != nil {
			return nil, nil, err
		}
		body = append(body, s)
	}
	p.next() // }
	return shared, body, nil
}

func (p *parser) block() ([]ir.Stmt, error) {
	if err := p.wantPunct("{"); err != nil {
		return nil, err
	}
	var body []ir.Stmt
	for !p.isPunct("}") {
		s, err := p.stmt()
		if err != nil {
			return nil, err
		}
		body = append(body, s)
	}
	p.next() // }
	return body, nil
}

func (p *parser) stmt() (ir.Stmt, error) {
	switch {
	case p.isIdent("let"):
		p.next()
		name, err := p.wantIdent()
		if err != nil {
			return nil, err
		}
		if err := p.wantPunct("="); err != nil {
			return nil, err
		}
		v, err := p.expr()
		if err != nil {
			return nil, err
		}
		if err := p.wantPunct(";"); err != nil {
			return nil, err
		}
		return ir.Let{Name: name, Value: v}, nil
	case p.isIdent("var"):
		v, err := p.varDecl()
		if err != nil {
			return nil, err
		}
		if err := p.wantPunct(";"); err != nil {
			return nil, err
		}
		return v, nil
	case p.isIdent("return"):
		p.next()
		if err := p.wantPunct(";"); err != nil {
			return nil, err
		}
		return ir.Return{}, nil
	case p.isIdent("break"):
		p.next()
		if err := p.wantPunct(";"); err != nil {
			return nil, err
		}
		return ir.Break{}, nil
	case p.isIdent("barrier"):
		p.next()
		if err := p.wantPunct(";"); err != nil {
			return nil, err
		}
		return ir.Barrier{}, nil
	case p.isIdent("if"):
		return p.ifStmt()
	case p.isIdent("for"):
		return p.forStmt()
	default:
		a, err := p.assignNoSemi()
		if err != nil {
			return nil, err
		}
		if err := p.wantPunct(";"); err != nil {
			return nil, err
		}
		return a, nil
	}
}

func (p *parser) varDecl() (ir.Var, error) {
	p.next() // var
	name, err := p.wantIdent()
	if err != nil {
		return ir.Var{}, err
	}
	var t ir.Type
	if p.isPunct(":") {
		p.next()
		if t, err = p.typ(); err != nil {
			return ir.Var{}, err
		}
	}
	if err := p.wantPunct("="); err != nil {
		return ir.Var{}, err
	}
	init, err := p.expr()
	if err != nil {
		return ir.Var{}, err
	}
	return ir.Var{Name: name, Type: t, Init: init}, nil
}

func (p *parser) assignNoSemi() (ir.Assign, error) {
	target, err := p.expr()
	if err != nil {
		return ir.Assign{}, err
	}
	op, err := p.assignOp()
	if err != nil {
		return ir.Assign{}, err
	}
	v, err := p.expr()
	if err != nil {
		return ir.Assign{}, err
	}
	return ir.Assign{Target: target, Value: v, Op: op}, nil
}

// assignOp consumes "=" (returning "") or a compound operator like "+="
// (returning "+"), matching ir.Assign.Op.
func (p *parser) assignOp() (string, error) {
	t := p.peek()
	if t.kind == tPunct {
		switch t.val {
		case "=":
			p.next()
			return "", nil
		case "+=", "-=", "*=", "/=", "%=":
			p.next()
			return strings.TrimSuffix(t.val, "="), nil
		}
	}
	return "", p.errf("expected assignment operator, got %q", t.val)
}

func (p *parser) ifStmt() (ir.Stmt, error) {
	p.next() // if
	cond, err := p.expr()
	if err != nil {
		return nil, err
	}
	then, err := p.block()
	if err != nil {
		return nil, err
	}
	var els []ir.Stmt
	if p.isIdent("else") {
		p.next()
		if p.isIdent("if") {
			s, err := p.ifStmt()
			if err != nil {
				return nil, err
			}
			els = []ir.Stmt{s}
		} else if els, err = p.block(); err != nil {
			return nil, err
		}
	}
	return ir.If{Cond: cond, Then: then, Else: els}, nil
}

func (p *parser) forStmt() (ir.Stmt, error) {
	p.next() // for
	if err := p.wantPunct("("); err != nil {
		return nil, err
	}
	var init ir.Stmt
	if p.isIdent("var") {
		v, err := p.varDecl()
		if err != nil {
			return nil, err
		}
		init = v
	} else {
		a, err := p.assignNoSemi()
		if err != nil {
			return nil, err
		}
		init = a
	}
	if err := p.wantPunct(";"); err != nil {
		return nil, err
	}
	cond, err := p.expr()
	if err != nil {
		return nil, err
	}
	if err := p.wantPunct(";"); err != nil {
		return nil, err
	}
	post, err := p.assignNoSemi()
	if err != nil {
		return nil, err
	}
	if err := p.wantPunct(")"); err != nil {
		return nil, err
	}
	body, err := p.block()
	if err != nil {
		return nil, err
	}
	return ir.For{Init: init, Cond: cond, Post: post, Body: body}, nil
}

// --- expressions (precedence climbing) ---

func (p *parser) expr() (ir.Expr, error) { return p.comparison() }

func (p *parser) comparison() (ir.Expr, error) {
	left, err := p.additive()
	if err != nil {
		return nil, err
	}
	for p.isCompareOp() {
		op := p.next().val
		right, err := p.additive()
		if err != nil {
			return nil, err
		}
		left = ir.Binary{Op: op, L: left, R: right}
	}
	return left, nil
}

func (p *parser) additive() (ir.Expr, error) {
	left, err := p.multiplicative()
	if err != nil {
		return nil, err
	}
	for p.isPunct("+") || p.isPunct("-") {
		op := p.next().val
		right, err := p.multiplicative()
		if err != nil {
			return nil, err
		}
		left = ir.Binary{Op: op, L: left, R: right}
	}
	return left, nil
}

func (p *parser) multiplicative() (ir.Expr, error) {
	left, err := p.unary()
	if err != nil {
		return nil, err
	}
	for p.isPunct("*") || p.isPunct("/") || p.isPunct("%") {
		op := p.next().val
		right, err := p.unary()
		if err != nil {
			return nil, err
		}
		left = ir.Binary{Op: op, L: left, R: right}
	}
	return left, nil
}

func (p *parser) unary() (ir.Expr, error) {
	if p.isPunct("-") || p.isPunct("!") || p.isPunct("&") {
		op := p.next().val
		e, err := p.unary()
		if err != nil {
			return nil, err
		}
		if op == "&" {
			return ir.AddrOf{E: e}, nil
		}
		return ir.Unary{Op: op, E: e}, nil
	}
	return p.postfix()
}

func (p *parser) postfix() (ir.Expr, error) {
	e, err := p.primary()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case p.isPunct("."):
			p.next()
			field, err := p.wantIdent()
			if err != nil {
				return nil, err
			}
			e = ir.Member{E: e, Field: field}
		case p.isPunct("["):
			p.next()
			idx, err := p.expr()
			if err != nil {
				return nil, err
			}
			if err := p.wantPunct("]"); err != nil {
				return nil, err
			}
			e = ir.Index{E: e, Idx: idx}
		default:
			return e, nil
		}
	}
}

func (p *parser) primary() (ir.Expr, error) {
	t := p.peek()
	switch {
	case t.kind == tNumber:
		p.next()
		return ir.Lit{Text: t.val}, nil
	case p.isIdent("true") || p.isIdent("false"):
		p.next()
		return ir.Lit{Text: t.val}, nil
	case t.kind == tIdent:
		p.next()
		if p.isPunct("(") {
			p.next()
			var args []ir.Expr
			for !p.isPunct(")") {
				a, err := p.expr()
				if err != nil {
					return nil, err
				}
				args = append(args, a)
				if p.isPunct(",") {
					p.next()
				}
			}
			p.next() // )
			return ir.Call{Func: t.val, Args: args}, nil
		}
		return ir.Name{Name: t.val}, nil
	case p.isPunct("("):
		p.next()
		e, err := p.expr()
		if err != nil {
			return nil, err
		}
		if err := p.wantPunct(")"); err != nil {
			return nil, err
		}
		return e, nil
	}
	return nil, p.errf("unexpected %q in expression", t.val)
}

func (p *parser) isCompareOp() bool {
	t := p.peek()
	if t.kind != tPunct {
		return false
	}
	switch t.val {
	case "<", ">", "<=", ">=", "==", "!=":
		return true
	}
	return false
}
