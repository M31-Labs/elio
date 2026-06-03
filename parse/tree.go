// Package parse is Elio's front-end: it parses .elio compute source into an
// ir.Module using the grammargen/gotreesitter grammar (grammar.ElioGrammar) —
// the house-style parser shared with Selena and Manta.
package parse

import (
	"fmt"
	"strconv"
	"strings"

	gts "github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/taproot"

	"m31labs.dev/elio/grammar"
	"m31labs.dev/elio/ir"
)

// Parse parses Elio source into an ir.Module using the grammargen-generated
// tree-sitter grammar (grammar.ElioGrammar).
func Parse(src string) (*ir.Module, error) {
	root, tw, err := taproot.Parse("elio", grammar.ElioGrammar, []byte(src))
	if err != nil {
		return nil, err
	}
	w := &treeWalker{Walker: tw}
	m := &ir.Module{}
	for i := 0; i < root.NamedChildCount(); i++ {
		c := root.NamedChild(i)
		switch w.Type(c) {
		case "struct_decl":
			s, err := w.structDecl(c)
			if err != nil {
				return nil, err
			}
			m.Structs = append(m.Structs, s)
		case "const_decl":
			t, err := w.typeOf(w.Field(c, "type"))
			if err != nil {
				return nil, err
			}
			v, err := w.expr(w.Field(c, "value"))
			if err != nil {
				return nil, err
			}
			m.Consts = append(m.Consts, ir.Const{Name: w.Text(w.Field(c, "name")), Type: t, Value: v})
		case "binding_decl":
			b, err := w.binding(c)
			if err != nil {
				return nil, err
			}
			m.Bindings = append(m.Bindings, b)
		case "kernel_decl":
			k, err := w.kernel(c)
			if err != nil {
				return nil, err
			}
			m.Kernels = append(m.Kernels, k)
		}
	}
	return m, nil
}

type treeWalker struct {
	*taproot.Walker
}

// span returns the 1-based source position where n begins.
func (w *treeWalker) span(n *gts.Node) ir.Span {
	line, col := w.Pos(n)
	return ir.Span{Line: line, Col: col}
}

func (w *treeWalker) structDecl(n *gts.Node) (ir.Struct, error) {
	s := ir.Struct{Name: w.Text(w.Field(n, "name"))}
	for i := 0; i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if w.Type(c) != "field_decl" {
			continue
		}
		t, err := w.typeOf(w.Field(c, "type"))
		if err != nil {
			return ir.Struct{}, err
		}
		s.Fields = append(s.Fields, ir.Field{Name: w.Text(w.Field(c, "name")), Type: t})
	}
	return s, nil
}

// scalarType maps an Elio type identifier to its ir.Type. Named types that are
// not known scalar/vector/matrix/atomic names are returned as ir.Named.
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

func (w *treeWalker) typeOf(n *gts.Node) (ir.Type, error) {
	switch w.Type(n) {
	case "type":
		// `type` is a plain (non-supertype) rule wrapping the concrete node.
		return w.typeOf(n.NamedChild(0))
	case "array_type":
		elem, err := w.typeOf(w.Field(n, "elem"))
		if err != nil {
			return nil, err
		}
		length := 0
		if l := w.Field(n, "len"); l != nil {
			length, _ = strconv.Atoi(w.Text(l))
		}
		return ir.Array{Elem: elem, Len: length}, nil
	case "identifier":
		return scalarType(w.Text(n)), nil
	}
	return nil, fmt.Errorf("parse: unexpected type node %q", w.Type(n))
}

func (w *treeWalker) binding(n *gts.Node) (ir.Binding, error) {
	g, _ := strconv.Atoi(w.Text(w.Field(n, "group")))
	b, _ := strconv.Atoi(w.Text(w.Field(n, "binding")))
	out := ir.Binding{Group: g, Binding: b, Name: w.Text(w.Field(n, "name"))}
	// qualifier text is "uniform" | "storage" | "storage read" | "storage read_write"
	fields := strings.Fields(w.Text(w.Field(n, "qualifier")))
	switch {
	case len(fields) > 0 && fields[0] == "uniform":
		out.Space = ir.Uniform
	case len(fields) > 0 && fields[0] == "storage":
		out.Space = ir.Storage
		if len(fields) > 1 {
			switch fields[1] {
			case "read":
				out.Access = ir.Read
			case "read_write":
				out.Access = ir.ReadWrite
			}
		}
	default:
		return ir.Binding{}, fmt.Errorf("parse: bad binding qualifier %q", w.Text(w.Field(n, "qualifier")))
	}
	t, err := w.typeOf(w.Field(n, "type"))
	if err != nil {
		return ir.Binding{}, err
	}
	out.Type = t
	return out, nil
}

func (w *treeWalker) kernel(n *gts.Node) (ir.Kernel, error) {
	k := ir.Kernel{Name: w.Text(w.Field(n, "name")), WorkgroupSize: [3]int{1, 1, 1}}
	for i, f := range []string{"wgx", "wgy", "wgz"} {
		if d := w.Field(n, f); d != nil {
			k.WorkgroupSize[i], _ = strconv.Atoi(w.Text(d))
		}
	}
	for i := 0; i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if w.Type(c) != "params" {
			continue
		}
		for j := 0; j < c.NamedChildCount(); j++ {
			p := c.NamedChild(j)
			if w.Type(p) != "param" {
				continue
			}
			k.Builtins = append(k.Builtins, ir.Builtin{
				Name:    w.Text(w.Field(p, "name")),
				Builtin: w.Text(w.Field(p, "builtin")),
				Type:    ir.Vec{N: 3, Elem: ir.U32},
			})
		}
	}
	// The kernel_body separates shared declarations (a prefix) from statements.
	bodyNode := w.Field(n, "body")
	for i := 0; i < bodyNode.NamedChildCount(); i++ {
		c := bodyNode.NamedChild(i)
		switch w.Type(c) {
		case "shared_decl":
			t, err := w.typeOf(w.Field(c, "type"))
			if err != nil {
				return ir.Kernel{}, err
			}
			k.Shared = append(k.Shared, ir.Shared{Name: w.Text(w.Field(c, "name")), Type: t})
		case "statement":
			st, err := w.stmt(c.NamedChild(0))
			if err != nil {
				return ir.Kernel{}, err
			}
			k.Body = append(k.Body, st)
		}
	}
	return k, nil
}

func (w *treeWalker) block(n *gts.Node) ([]ir.Stmt, error) {
	var body []ir.Stmt
	for i := 0; i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if w.Type(c) != "statement" {
			// supertype may expose the concrete statement directly
			if s, ok := w.maybeStmt(c); ok {
				st, err := s()
				if err != nil {
					return nil, err
				}
				body = append(body, st)
			}
			continue
		}
		st, err := w.stmt(c.NamedChild(0))
		if err != nil {
			return nil, err
		}
		body = append(body, st)
	}
	return body, nil
}

// maybeStmt returns a thunk if n is a concrete statement node (used when the
// `statement` supertype is elided and the concrete node appears directly).
func (w *treeWalker) maybeStmt(n *gts.Node) (func() (ir.Stmt, error), bool) {
	switch w.Type(n) {
	case "let_stmt", "var_stmt", "return_stmt", "break_stmt", "barrier_stmt", "if_stmt", "for_stmt", "while_stmt", "assign_stmt":
		return func() (ir.Stmt, error) { return w.stmt(n) }, true
	}
	return nil, false
}

func (w *treeWalker) stmt(n *gts.Node) (ir.Stmt, error) {
	switch w.Type(n) {
	case "let_stmt":
		v, err := w.expr(w.Field(n, "value"))
		if err != nil {
			return nil, err
		}
		return ir.Let{Name: w.Text(w.Field(n, "name")), Value: v, Span: w.span(n)}, nil
	case "var_stmt":
		return w.varInner(n.NamedChild(0))
	case "return_stmt":
		return ir.Return{Span: w.span(n)}, nil
	case "break_stmt":
		return ir.Break{Span: w.span(n)}, nil
	case "barrier_stmt":
		return ir.Barrier{Span: w.span(n)}, nil
	case "if_stmt":
		return w.ifStmt(n)
	case "for_stmt":
		return w.forStmt(n)
	case "while_stmt":
		cond, err := w.expr(w.Field(n, "cond"))
		if err != nil {
			return nil, err
		}
		body, err := w.block(w.Field(n, "body"))
		if err != nil {
			return nil, err
		}
		return ir.While{Cond: cond, Body: body, Span: w.span(n)}, nil
	case "assign_stmt":
		return w.assignInner(n.NamedChild(0))
	}
	return nil, fmt.Errorf("parse: unexpected statement node %q", w.Type(n))
}

func (w *treeWalker) varInner(n *gts.Node) (ir.Stmt, error) {
	v := ir.Var{Name: w.Text(w.Field(n, "name")), Span: w.span(n)}
	if t := w.Field(n, "type"); t != nil {
		typ, err := w.typeOf(t)
		if err != nil {
			return nil, err
		}
		v.Type = typ
	}
	init, err := w.expr(w.Field(n, "value"))
	if err != nil {
		return nil, err
	}
	v.Init = init
	return v, nil
}

func (w *treeWalker) assignInner(n *gts.Node) (ir.Stmt, error) {
	target, err := w.expr(w.Field(n, "target"))
	if err != nil {
		return nil, err
	}
	value, err := w.expr(w.Field(n, "value"))
	if err != nil {
		return nil, err
	}
	return ir.Assign{Target: target, Value: value, Op: compoundOp(w.Text(w.Field(n, "op"))), Span: w.span(n)}, nil
}

// compoundOp maps an assignment operator token to ir.Assign.Op: "=" → "" (plain
// assign), "+=" → "+", and so on.
func compoundOp(tok string) string {
	if tok == "=" || tok == "" {
		return ""
	}
	return strings.TrimSuffix(tok, "=")
}

func (w *treeWalker) ifStmt(n *gts.Node) (ir.Stmt, error) {
	cond, err := w.expr(w.Field(n, "cond"))
	if err != nil {
		return nil, err
	}
	then, err := w.block(w.Field(n, "then"))
	if err != nil {
		return nil, err
	}
	out := ir.If{Cond: cond, Then: then, Span: w.span(n)}
	if e := w.Field(n, "else"); e != nil {
		switch w.Type(e) {
		case "if_stmt":
			s, err := w.ifStmt(e)
			if err != nil {
				return nil, err
			}
			out.Else = []ir.Stmt{s}
		case "block":
			els, err := w.block(e)
			if err != nil {
				return nil, err
			}
			out.Else = els
		}
	}
	return out, nil
}

func (w *treeWalker) forStmt(n *gts.Node) (ir.Stmt, error) {
	initNode := w.Field(n, "init")
	var init ir.Stmt
	var err error
	switch w.Type(initNode) {
	case "var_inner":
		init, err = w.varInner(initNode)
	case "assign_inner":
		init, err = w.assignInner(initNode)
	default:
		return nil, fmt.Errorf("parse: unexpected for-init node %q", w.Type(initNode))
	}
	if err != nil {
		return nil, err
	}
	cond, err := w.expr(w.Field(n, "cond"))
	if err != nil {
		return nil, err
	}
	post, err := w.assignInner(w.Field(n, "post"))
	if err != nil {
		return nil, err
	}
	body, err := w.block(w.Field(n, "body"))
	if err != nil {
		return nil, err
	}
	return ir.For{Init: init, Cond: cond, Post: post, Body: body, Span: w.span(n)}, nil
}

func (w *treeWalker) expr(n *gts.Node) (ir.Expr, error) {
	switch w.Type(n) {
	case "expression", "postfix_expression", "primary_expression":
		// supertype wrapper materialized — unwrap to the concrete node
		return w.expr(n.NamedChild(0))
	case "number":
		return ir.Lit{Text: w.Text(n)}, nil
	case "identifier":
		t := w.Text(n)
		if t == "true" || t == "false" {
			return ir.Lit{Text: t}, nil
		}
		return ir.Name{Name: t}, nil
	case "paren_expression":
		return w.expr(n.NamedChild(0))
	case "binary_expression":
		l, err := w.expr(w.Field(n, "left"))
		if err != nil {
			return nil, err
		}
		r, err := w.expr(w.Field(n, "right"))
		if err != nil {
			return nil, err
		}
		return ir.Binary{Op: w.Text(w.Field(n, "operator")), L: l, R: r}, nil
	case "unary_expression":
		e, err := w.expr(w.Field(n, "operand"))
		if err != nil {
			return nil, err
		}
		if w.Text(w.Field(n, "op")) == "&" {
			return ir.AddrOf{E: e}, nil
		}
		return ir.Unary{Op: w.Text(w.Field(n, "op")), E: e}, nil
	case "member_expression":
		obj, err := w.expr(w.Field(n, "object"))
		if err != nil {
			return nil, err
		}
		return ir.Member{E: obj, Field: w.Text(w.Field(n, "field"))}, nil
	case "index_expression":
		obj, err := w.expr(w.Field(n, "object"))
		if err != nil {
			return nil, err
		}
		idx, err := w.expr(w.Field(n, "index"))
		if err != nil {
			return nil, err
		}
		return ir.Index{E: obj, Idx: idx}, nil
	case "call":
		var args []ir.Expr
		for i := 0; i < n.NamedChildCount(); i++ {
			c := n.NamedChild(i)
			if w.Type(c) != "arguments" {
				continue
			}
			for j := 0; j < c.NamedChildCount(); j++ {
				a, err := w.expr(c.NamedChild(j))
				if err != nil {
					return nil, err
				}
				args = append(args, a)
			}
		}
		return ir.Call{Func: w.Text(w.Field(n, "callee")), Args: args}, nil
	}
	return nil, fmt.Errorf("parse: unexpected expression node %q", w.Type(n))
}
