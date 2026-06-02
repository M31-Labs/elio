package parse

import (
	"os"
	"path/filepath"
	"testing"

	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
)

// TestParseTreeMatchesHandWritten is the grammargen front-end proof: the
// tree-sitter parser (grammar.ElioGrammar → ParseTree) must produce IR that
// emits byte-identical WGSL to the hand-written recursive-descent parser. This
// pins the postfix-chain grammar fix end-to-end: source → tree → IR → WGSL lands
// exactly where the reference parser does.
func TestParseTreeMatchesHandWritten(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "cull.elio"))
	if err != nil {
		t.Fatal(err)
	}
	fromTree, err := ParseTree(string(src))
	if err != nil {
		t.Fatalf("ParseTree: %v", err)
	}
	fromHand, err := Parse(string(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	treeWGSL, err := wgsl.Emit(fromTree)
	if err != nil {
		t.Fatalf("emit tree WGSL: %v", err)
	}
	handWGSL, err := wgsl.Emit(fromHand)
	if err != nil {
		t.Fatalf("emit hand WGSL: %v", err)
	}
	if treeWGSL != handWGSL {
		t.Errorf("grammargen parse emits differently than hand-written\n--- tree ---\n%s\n--- hand ---\n%s", treeWGSL, handWGSL)
	}
}

// TestParseTreeReduceEndToEnd pins the workgroup-shared / barrier surface: both
// parsers must turn reduce.elio into IR that emits identical WGSL to the
// hand-built ir.WorkgroupReduce() — proving `shared`/`barrier` author end-to-end.
func TestParseTreeReduceEndToEnd(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "reduce.elio"))
	if err != nil {
		t.Fatal(err)
	}
	fromTree, err := ParseTree(string(src))
	if err != nil {
		t.Fatalf("ParseTree: %v", err)
	}
	fromHand, err := Parse(string(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	treeWGSL, _ := wgsl.Emit(fromTree)
	handWGSL, _ := wgsl.Emit(fromHand)
	wantWGSL, _ := wgsl.Emit(ir.WorkgroupReduce())
	if treeWGSL != wantWGSL {
		t.Errorf("grammargen parse of reduce.elio != hand-built IR\n--- tree ---\n%s\n--- want ---\n%s", treeWGSL, wantWGSL)
	}
	if handWGSL != wantWGSL {
		t.Errorf("hand-written parse of reduce.elio != hand-built IR\n--- hand ---\n%s\n--- want ---\n%s", handWGSL, wantWGSL)
	}
}

// TestParseTreeMatchesGolden pins ParseTree against the naga-validated WGSL
// golden directly (the same golden the hand-written parser is held to).
func TestParseTreeMatchesGolden(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "cull.elio"))
	if err != nil {
		t.Fatal(err)
	}
	mod, err := ParseTree(string(src))
	if err != nil {
		t.Fatalf("ParseTree: %v", err)
	}
	got, err := wgsl.Emit(mod)
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	want, err := os.ReadFile(filepath.Join("..", "emit", "wgsl", "testdata", "cull.wgsl"))
	if err != nil {
		t.Fatalf("read wgsl golden: %v", err)
	}
	if got != string(want) {
		t.Errorf("ParseTree-then-emitted WGSL != golden\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

// TestParseTreeRejectsOutermostIndexRegression is the direct regression for the
// grammargen LALR-merge bug: index_expression as the statement-final expression
// (`let r = a[b];`) must parse. Before the postfix-chain restructure, the full
// grammar dropped this reduce action and the parse failed.
func TestParseTreeRejectsOutermostIndexRegression(t *testing.T) {
	cases := []string{
		"@workgroup(64) kernel main(gid: x) {\n  let record = input[i];\n}\n",
		"@workgroup(64) kernel main(gid: x) {\n  out[s] = record;\n}\n",
		"@workgroup(64) kernel main(gid: x) {\n  let c = record.model[3];\n}\n",
		"@workgroup(64) kernel main(gid: x) {\n  let s = atomicAdd(&drawArgs[1], 1u);\n}\n",
	}
	for _, src := range cases {
		if _, err := ParseTree(src); err != nil {
			t.Errorf("ParseTree(%q): %v", src, err)
		}
	}
}
