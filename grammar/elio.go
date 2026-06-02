// Package grammar defines Elio's .elio compute language with grammargen — the
// same pure-Go tree-sitter engine behind Selena's .sel and GoSX's .gsx, so the
// triad shares one parser toolchain.
//
// Surface (Go-flavored types avoid WGSL's <…> generic ambiguity):
//
//	struct Name { field: type; … }
//	@group(g) @binding(b) uniform|storage [read|read_write] name: T;
//	@workgroup(x[,y,z]) kernel name(p: builtin, …) { stmts }
//
// # Expression structure: primary + postfix chain (NOT supertype left-recursion)
//
// member/index/call are a tight postfix chain over primary_expression, rather
// than being left-recursive on the full `expression` supertype. That is the
// canonical tree-sitter idiom, and here it is load-bearing: when the postfix
// operators recurse on `expression` AND the grammar carries three or more unary
// prefix operators (Elio has -, !, &), grammargen's LALR state-merge crosses a
// threshold where it silently drops the reduce action for a postfix operator in
// statement-final position — so `let r = a[b];` fails to parse while
// `let r = a[b] + c;` (index not outermost) succeeds. The postfix-chain shape
// keeps each postfix reduce in a cleanly-separable state and sidesteps the
// defect. See grammar/elio_test.go for the regression that pins this.
package grammar

import "github.com/odvcencio/gotreesitter/grammargen"

// ElioGrammar returns the grammargen grammar for the .elio compute language.
func ElioGrammar() *grammargen.Grammar {
	s := grammargen.Str
	sym := grammargen.Sym
	seq := grammargen.Seq
	field := grammargen.Field
	choice := grammargen.Choice
	opt := grammargen.Optional
	rep := grammargen.Repeat

	g := grammargen.NewGrammar("elio")

	g.Define("source_file", rep(choice(
		sym("struct_decl"), sym("binding_decl"), sym("kernel_decl"),
	)))

	// struct Name { field: type; … }
	g.Define("struct_decl", seq(
		s("struct"), field("name", sym("identifier")),
		s("{"), rep(sym("field_decl")), s("}"),
	))
	g.Define("field_decl", seq(
		field("name", sym("identifier")), s(":"), field("type", sym("type")), s(";"),
	))

	// @group(g) @binding(b) <qualifier> name : type ;
	g.Define("binding_decl", seq(
		s("@"), s("group"), s("("), field("group", sym("number")), s(")"),
		s("@"), s("binding"), s("("), field("binding", sym("number")), s(")"),
		field("qualifier", sym("binding_qualifier")),
		field("name", sym("identifier")), s(":"), field("type", sym("type")), s(";"),
	))
	g.Define("binding_qualifier", choice(
		s("uniform"),
		seq(s("storage"), opt(choice(s("read"), s("read_write")))),
	))

	// @workgroup(x[,y,z]) kernel name(param: builtin, …) { body }
	g.Define("kernel_decl", seq(
		s("@"), s("workgroup"), s("("),
		field("wgx", sym("number")),
		opt(seq(s(","), field("wgy", sym("number")))),
		opt(seq(s(","), field("wgz", sym("number")))),
		s(")"),
		s("kernel"), field("name", sym("identifier")),
		s("("), opt(sym("params")), s(")"),
		field("body", sym("kernel_body")),
	))
	// A kernel body is workgroup-shared declarations (a prefix) then statements.
	g.Define("kernel_body", seq(s("{"), rep(sym("shared_decl")), rep(sym("statement")), s("}")))
	g.Define("shared_decl", seq(
		s("shared"), field("name", sym("identifier")), s(":"), field("type", sym("type")), s(";"),
	))
	g.Define("params", seq(sym("param"), rep(seq(s(","), sym("param"))), opt(s(","))))
	g.Define("param", seq(
		field("name", sym("identifier")), s(":"), field("builtin", sym("identifier")),
	))

	// Types: [N]T fixed array · []T runtime array · scalar/vec/mat/atomic/named.
	g.Define("type", choice(sym("array_type"), sym("identifier")))
	g.Define("array_type", seq(
		s("["), opt(field("len", sym("number"))), s("]"), field("elem", sym("type")),
	))

	g.Define("block", seq(s("{"), rep(sym("statement")), s("}")))

	g.Define("statement", choice(
		sym("let_stmt"), sym("var_stmt"), sym("return_stmt"),
		sym("break_stmt"), sym("barrier_stmt"), sym("if_stmt"), sym("for_stmt"), sym("assign_stmt"),
	))

	g.Define("let_stmt", seq(
		s("let"), field("name", sym("identifier")), s("="), field("value", sym("expression")), s(";"),
	))
	g.Define("var_stmt", seq(sym("var_inner"), s(";")))
	g.Define("var_inner", seq(
		s("var"), field("name", sym("identifier")),
		opt(seq(s(":"), field("type", sym("type")))),
		s("="), field("value", sym("expression")),
	))
	g.Define("return_stmt", seq(s("return"), s(";")))
	g.Define("break_stmt", seq(s("break"), s(";")))
	g.Define("barrier_stmt", seq(s("barrier"), s(";")))
	g.Define("if_stmt", seq(
		s("if"), field("cond", sym("expression")), field("then", sym("block")),
		opt(seq(s("else"), field("else", choice(sym("if_stmt"), sym("block"))))),
	))
	g.Define("for_stmt", seq(
		s("for"), s("("),
		field("init", choice(sym("var_inner"), sym("assign_inner"))), s(";"),
		field("cond", sym("expression")), s(";"),
		field("post", sym("assign_inner")), s(")"),
		field("body", sym("block")),
	))
	g.Define("assign_stmt", seq(sym("assign_inner"), s(";")))
	g.Define("assign_inner", seq(
		field("target", sym("expression")),
		field("op", sym("assign_op")),
		field("value", sym("expression")),
	))
	// One lexed token for "=" and the compound forms, rather than a 6-way choice
	// of string literals — the choice grows the parser state enough to tip a
	// reduce over grammargen's LALR-merge threshold (see package doc).
	g.Define("assign_op", grammargen.Token(grammargen.Pat(`[-+*/%]?=`)))

	// --- expression tower: binary > unary > postfix > primary ---
	g.Define("expression", choice(
		sym("binary_expression"),
		sym("unary_expression"),
		sym("postfix_expression"),
		sym("primary_expression"),
	))

	g.Define("binary_expression", choice(
		grammargen.PrecLeft(0, seq(field("left", sym("expression")),
			field("operator", choice(s("<"), s(">"), s("<="), s(">="), s("=="), s("!="))),
			field("right", sym("expression")))),
		grammargen.PrecLeft(1, seq(field("left", sym("expression")),
			field("operator", choice(s("+"), s("-"))),
			field("right", sym("expression")))),
		grammargen.PrecLeft(2, seq(field("left", sym("expression")),
			field("operator", choice(s("*"), s("/"), s("%"))),
			field("right", sym("expression")))),
	))

	// Prefix -, !, & over a full-expression operand. PrecRight(4): binds looser
	// than the postfix operators (5), so -a[b] is -(a[b]).
	g.Define("unary_expression", grammargen.PrecRight(4, seq(
		field("op", choice(s("-"), s("!"), s("&"))),
		field("operand", sym("expression")),
	)))

	// Postfix chain: member/index/call recurse on postfix_expression (the FIX —
	// see package doc). object/callee are postfix_expression|primary_expression.
	g.Define("postfix_expression", choice(
		sym("member_expression"), sym("index_expression"), sym("call"),
	))
	postfixBase := choice(sym("postfix_expression"), sym("primary_expression"))
	g.Define("member_expression", grammargen.PrecLeft(5, seq(
		field("object", postfixBase), s("."), field("field", sym("identifier")),
	)))
	g.Define("index_expression", grammargen.PrecLeft(5, seq(
		field("object", postfixBase), s("["), field("index", sym("expression")), s("]"),
	)))
	g.Define("call", grammargen.PrecLeft(5, seq(
		field("callee", sym("identifier")), s("("), opt(sym("arguments")), s(")"),
	)))
	g.Define("arguments", seq(sym("expression"), rep(seq(s(","), sym("expression"))), opt(s(","))))

	g.Define("primary_expression", choice(
		sym("paren_expression"), sym("number"), sym("identifier"),
	))
	g.Define("paren_expression", seq(s("("), sym("expression"), s(")")))

	g.Define("identifier", grammargen.Token(grammargen.Pat(`[A-Za-z_][A-Za-z0-9_]*`)))
	g.Define("number", grammargen.Token(grammargen.Pat(`[0-9]+(\.[0-9]+)?[uif]?`)))

	g.SetWord("identifier")
	g.SetSupertypes("expression", "postfix_expression", "primary_expression")
	g.SetExtras(grammargen.Pat(`\s`), grammargen.Pat(`//[^\n]*`))

	return g
}
