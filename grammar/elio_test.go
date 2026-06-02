package grammar

import (
	"testing"

	gts "github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/grammargen"
)

func mustLang(t *testing.T) *gts.Language {
	t.Helper()
	lang, _, err := grammargen.GenerateLanguageAndBlob(ElioGrammar())
	if err != nil {
		t.Fatalf("generate elio language: %v", err)
	}
	return lang
}

func parseOK(t *testing.T, lang *gts.Language, src string) bool {
	t.Helper()
	tree, err := gts.NewParser(lang).Parse([]byte(src))
	if err != nil || tree == nil || tree.RootNode() == nil {
		return false
	}
	return !tree.RootNode().HasError()
}

// TestOutermostIndexParses is the direct grammar-level regression for the
// grammargen LALR-merge defect documented in elio.go: an index_expression as
// the statement-final expression must parse. Under the previous
// supertype-left-recursion shape (member/index/call recursive on `expression`)
// plus three unary prefix operators, grammargen dropped the index reduce action
// in statement-final position and these failed. The postfix-chain shape fixes it.
func TestOutermostIndexParses(t *testing.T) {
	lang := mustLang(t)
	K := func(body string) string { return "@workgroup(64) kernel main(gid: x) {\n  " + body + "\n}\n" }
	mustParse := []string{
		K("let record = input[i];"),     // index as let value (the core failure)
		K("let record = input[0];"),      // numeric index
		K("out[s] = record;"),            // index as assign target
		K("let c = record.model[3];"),    // index on a member, outermost
		K("let c = record.model[3].xyz;"),// member-index-member chain
		K("let r = input[i] + a;"),       // index not outermost (always worked)
		K("let u = -a[b];"),              // unary over index
		K("let s = atomicAdd(&drawArgs[1], 1u);"), // index inside call arg with addr-of
	}
	for _, src := range mustParse {
		if !parseOK(t, lang, src) {
			t.Errorf("expected clean parse:\n%s", src)
		}
	}
}

// TestGrammarGenerates pins that the grammar compiles to a language without
// error (a guard against future edits reintroducing a generation failure).
func TestGrammarGenerates(t *testing.T) {
	mustLang(t)
}
