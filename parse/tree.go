// tree.go is Elio's grammargen/gotreesitter front-end: it generates the .elio
// tree-sitter language from grammar.ElioGrammar and walks the parse tree into an
// ir.Module. This is the house-style parser (the same toolchain Selena and Manta
// use); the hand-written parser in parse.go remains as a reference oracle, and
// tree_test.go pins that both produce identical IR.
package parse

import (
	"fmt"
	"strconv"
	"strings"
	"sync"

	gts "github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/grammargen"

	"m31labs.dev/elio/grammar"
	"m31labs.dev/elio/ir"
)

var (
	elioLangOnce sync.Once
	elioLang     *gts.Language
	elioLangErr  error
)

func elioLanguage() (*gts.Language, error) {
	elioLangOnce.Do(func() {
		elioLang, _, elioLangErr = grammargen.GenerateLanguageAndBlob(grammar.ElioGrammar())
	})
	return elioLang, elioLangErr
}

// ParseTree parses Elio source into an ir.Module using the grammargen-generated
// tree-sitter grammar (grammar.ElioGrammar).
func ParseTree(src string) (*ir.Module, error) {
	l, err := elioLanguage()
	if err != nil {
		return nil, fmt.Errorf("generate elio language: %w", err)
	}
	tree, err := gts.NewParser(l).Parse([]byte(src))
	if err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	root := tree.RootNode()
	w := &treeWalker{lang: l, src: []byte(src)}
	if root.HasError() {
		return nil, w.syntaxError(root)
	}
	m := &ir.Module{}
	for i := 0; i < root.NamedChildCount(); i++ {
		c := root.NamedChild(i)
		switch w.typ(c) {
		case "struct_decl":
			s, err := w.structDecl(c)
			if err != nil {
				return nil, err
			}
			m.Structs = append(m.Structs, s)
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
	lang *gts.Language
	src  []byte
}

func (w *treeWalker) typ(n *gts.Node) string                { return n.Type(w.lang) }
func (w *treeWalker) text(n *gts.Node) string               { return n.Text(w.src) }
func (w *treeWalker) field(n *gts.Node, f string) *gts.Node { return n.ChildByFieldName(f, w.lang) }

func (w *treeWalker) syntaxError(n *gts.Node) error {
	// Find the deepest error/missing node for a precise message.
	var find func(n *gts.Node) *gts.Node
	find = func(n *gts.Node) *gts.Node {
		for i := 0; i < n.ChildCount(); i++ {
			c := n.Child(i)
			if c == nil {
				continue
			}
			if c.Type(w.lang) == "ERROR" || c.IsError() || c.IsMissing() {
				if deeper := find(c); deeper != nil {
					return deeper
				}
				return c
			}
			if deeper := find(c); deeper != nil {
				return deeper
			}
		}
		return nil
	}
	bad := find(n)
	if bad == nil {
		return fmt.Errorf("parse: syntax error")
	}
	near := strings.TrimSpace(w.text(bad))
	if len(near) > 30 {
		near = near[:30] + "…"
	}
	return fmt.Errorf("parse: syntax error at byte %d near %q", bad.StartByte(), near)
}

func (w *treeWalker) structDecl(n *gts.Node) (ir.Struct, error) {
	s := ir.Struct{Name: w.text(w.field(n, "name"))}
	for i := 0; i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if w.typ(c) != "field_decl" {
			continue
		}
		t, err := w.typeOf(w.field(c, "type"))
		if err != nil {
			return ir.Struct{}, err
		}
		s.Fields = append(s.Fields, ir.Field{Name: w.text(w.field(c, "name")), Type: t})
	}
	return s, nil
}

func (w *treeWalker) typeOf(n *gts.Node) (ir.Type, error) {
	switch w.typ(n) {
	case "type":
		// `type` is a plain (non-supertype) rule wrapping the concrete node.
		return w.typeOf(n.NamedChild(0))
	case "array_type":
		elem, err := w.typeOf(w.field(n, "elem"))
		if err != nil {
			return nil, err
		}
		length := 0
		if l := w.field(n, "len"); l != nil {
			length, _ = strconv.Atoi(w.text(l))
		}
		return ir.Array{Elem: elem, Len: length}, nil
	case "identifier":
		return scalarType(w.text(n)), nil
	}
	return nil, fmt.Errorf("parse: unexpected type node %q", w.typ(n))
}

func (w *treeWalker) binding(n *gts.Node) (ir.Binding, error) {
	g, _ := strconv.Atoi(w.text(w.field(n, "group")))
	b, _ := strconv.Atoi(w.text(w.field(n, "binding")))
	out := ir.Binding{Group: g, Binding: b, Name: w.text(w.field(n, "name"))}
	// qualifier text is "uniform" | "storage" | "storage read" | "storage read_write"
	fields := strings.Fields(w.text(w.field(n, "qualifier")))
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
		return ir.Binding{}, fmt.Errorf("parse: bad binding qualifier %q", w.text(w.field(n, "qualifier")))
	}
	t, err := w.typeOf(w.field(n, "type"))
	if err != nil {
		return ir.Binding{}, err
	}
	out.Type = t
	return out, nil
}

func (w *treeWalker) kernel(n *gts.Node) (ir.Kernel, error) {
	k := ir.Kernel{Name: w.text(w.field(n, "name")), WorkgroupSize: [3]int{1, 1, 1}}
	for i, f := range []string{"wgx", "wgy", "wgz"} {
		if d := w.field(n, f); d != nil {
			k.WorkgroupSize[i], _ = strconv.Atoi(w.text(d))
		}
	}
	for i := 0; i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if w.typ(c) != "params" {
			continue
		}
		for j := 0; j < c.NamedChildCount(); j++ {
			p := c.NamedChild(j)
			if w.typ(p) != "param" {
				continue
			}
			k.Builtins = append(k.Builtins, ir.Builtin{
				Name:    w.text(w.field(p, "name")),
				Builtin: w.text(w.field(p, "builtin")),
				Type:    ir.Vec{N: 3, Elem: ir.U32},
			})
		}
	}
	// The kernel_body separates shared declarations (a prefix) from statements.
	bodyNode := w.field(n, "body")
	for i := 0; i < bodyNode.NamedChildCount(); i++ {
		c := bodyNode.NamedChild(i)
		switch w.typ(c) {
		case "shared_decl":
			t, err := w.typeOf(w.field(c, "type"))
			if err != nil {
				return ir.Kernel{}, err
			}
			k.Shared = append(k.Shared, ir.Shared{Name: w.text(w.field(c, "name")), Type: t})
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
		if w.typ(c) != "statement" {
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
	switch w.typ(n) {
	case "let_stmt", "var_stmt", "return_stmt", "break_stmt", "barrier_stmt", "if_stmt", "for_stmt", "assign_stmt":
		return func() (ir.Stmt, error) { return w.stmt(n) }, true
	}
	return nil, false
}

func (w *treeWalker) stmt(n *gts.Node) (ir.Stmt, error) {
	switch w.typ(n) {
	case "let_stmt":
		v, err := w.expr(w.field(n, "value"))
		if err != nil {
			return nil, err
		}
		return ir.Let{Name: w.text(w.field(n, "name")), Value: v}, nil
	case "var_stmt":
		return w.varInner(n.NamedChild(0))
	case "return_stmt":
		return ir.Return{}, nil
	case "break_stmt":
		return ir.Break{}, nil
	case "barrier_stmt":
		return ir.Barrier{}, nil
	case "if_stmt":
		return w.ifStmt(n)
	case "for_stmt":
		return w.forStmt(n)
	case "assign_stmt":
		return w.assignInner(n.NamedChild(0))
	}
	return nil, fmt.Errorf("parse: unexpected statement node %q", w.typ(n))
}

func (w *treeWalker) varInner(n *gts.Node) (ir.Stmt, error) {
	v := ir.Var{Name: w.text(w.field(n, "name"))}
	if t := w.field(n, "type"); t != nil {
		typ, err := w.typeOf(t)
		if err != nil {
			return nil, err
		}
		v.Type = typ
	}
	init, err := w.expr(w.field(n, "value"))
	if err != nil {
		return nil, err
	}
	v.Init = init
	return v, nil
}

func (w *treeWalker) assignInner(n *gts.Node) (ir.Stmt, error) {
	target, err := w.expr(w.field(n, "target"))
	if err != nil {
		return nil, err
	}
	value, err := w.expr(w.field(n, "value"))
	if err != nil {
		return nil, err
	}
	return ir.Assign{Target: target, Value: value}, nil
}

func (w *treeWalker) ifStmt(n *gts.Node) (ir.Stmt, error) {
	cond, err := w.expr(w.field(n, "cond"))
	if err != nil {
		return nil, err
	}
	then, err := w.block(w.field(n, "then"))
	if err != nil {
		return nil, err
	}
	out := ir.If{Cond: cond, Then: then}
	if e := w.field(n, "else"); e != nil {
		switch w.typ(e) {
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
	initNode := w.field(n, "init")
	var init ir.Stmt
	var err error
	switch w.typ(initNode) {
	case "var_inner":
		init, err = w.varInner(initNode)
	case "assign_inner":
		init, err = w.assignInner(initNode)
	default:
		return nil, fmt.Errorf("parse: unexpected for-init node %q", w.typ(initNode))
	}
	if err != nil {
		return nil, err
	}
	cond, err := w.expr(w.field(n, "cond"))
	if err != nil {
		return nil, err
	}
	post, err := w.assignInner(w.field(n, "post"))
	if err != nil {
		return nil, err
	}
	body, err := w.block(w.field(n, "body"))
	if err != nil {
		return nil, err
	}
	return ir.For{Init: init, Cond: cond, Post: post, Body: body}, nil
}

func (w *treeWalker) expr(n *gts.Node) (ir.Expr, error) {
	switch w.typ(n) {
	case "expression", "postfix_expression", "primary_expression":
		// supertype wrapper materialized — unwrap to the concrete node
		return w.expr(n.NamedChild(0))
	case "number":
		return ir.Lit{Text: w.text(n)}, nil
	case "identifier":
		t := w.text(n)
		if t == "true" || t == "false" {
			return ir.Lit{Text: t}, nil
		}
		return ir.Name{Name: t}, nil
	case "paren_expression":
		return w.expr(n.NamedChild(0))
	case "binary_expression":
		l, err := w.expr(w.field(n, "left"))
		if err != nil {
			return nil, err
		}
		r, err := w.expr(w.field(n, "right"))
		if err != nil {
			return nil, err
		}
		return ir.Binary{Op: w.text(w.field(n, "operator")), L: l, R: r}, nil
	case "unary_expression":
		e, err := w.expr(w.field(n, "operand"))
		if err != nil {
			return nil, err
		}
		if w.text(w.field(n, "op")) == "&" {
			return ir.AddrOf{E: e}, nil
		}
		return ir.Unary{Op: w.text(w.field(n, "op")), E: e}, nil
	case "member_expression":
		obj, err := w.expr(w.field(n, "object"))
		if err != nil {
			return nil, err
		}
		return ir.Member{E: obj, Field: w.text(w.field(n, "field"))}, nil
	case "index_expression":
		obj, err := w.expr(w.field(n, "object"))
		if err != nil {
			return nil, err
		}
		idx, err := w.expr(w.field(n, "index"))
		if err != nil {
			return nil, err
		}
		return ir.Index{E: obj, Idx: idx}, nil
	case "call":
		var args []ir.Expr
		for i := 0; i < n.NamedChildCount(); i++ {
			c := n.NamedChild(i)
			if w.typ(c) != "arguments" {
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
		return ir.Call{Func: w.text(w.field(n, "callee")), Args: args}, nil
	}
	return nil, fmt.Errorf("parse: unexpected expression node %q", w.typ(n))
}
