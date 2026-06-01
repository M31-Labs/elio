package parse

import (
	"os"
	"path/filepath"
	"testing"

	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
)

// TestParseCullMatchesGolden is the front-end proof: parse cull.elio into IR,
// emit WGSL, and assert it is byte-identical to the WGSL backend's golden (which
// is itself naga-validated and generated from the hand-built ir.CullKernel()).
// Source → IR → WGSL therefore lands exactly where the hand-built IR does.
func TestParseCullMatchesGolden(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "cull.elio"))
	if err != nil {
		t.Fatal(err)
	}
	mod, err := Parse(string(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
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
		t.Errorf("parsed-then-emitted WGSL != hand-built golden\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

// TestParseEqualsHandBuilt pins that the parsed module is structurally identical
// to the hand-built one (same emitted WGSL from both paths).
func TestParseEqualsHandBuilt(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "cull.elio"))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(string(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	fromParse, _ := wgsl.Emit(parsed)
	fromHand, _ := wgsl.Emit(ir.CullKernel())
	if fromParse != fromHand {
		t.Errorf("parsed IR emits differently than hand-built IR\n--- parsed ---\n%s\n--- hand ---\n%s", fromParse, fromHand)
	}
}
