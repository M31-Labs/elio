package parse

import (
	"bytes"
	"testing"

	"github.com/odvcencio/gotreesitter/grammargen"
	"github.com/odvcencio/gotreesitter/taproot"

	"m31labs.dev/elio/grammar"
)

func TestGrammarBinIsCurrent(t *testing.T) {
	_, fresh, err := grammargen.GenerateLanguageAndBlob(grammar.ElioGrammar())
	if err != nil {
		t.Fatalf("GenerateLanguageAndBlob: %v", err)
	}
	if !bytes.Equal(fresh, grammarBlob) {
		t.Fatalf("grammar.bin is stale (embedded %d bytes, regenerated %d bytes); run `go generate ./...`",
			len(grammarBlob), len(fresh))
	}
}

func TestEmbeddedGrammarBlobParses(t *testing.T) {
	if len(grammarBlob) == 0 {
		t.Fatal("grammar.bin embed is empty")
	}
	src := []byte("@workgroup(64) kernel main(gid: x) {\n  let record = input[i];\n}\n")
	root, _, err := taproot.ParseFromBlob("elio-blob-test", grammarBlob, nil, src)
	if err != nil {
		t.Fatalf("ParseFromBlob: %v", err)
	}
	if root == nil || root.HasError() {
		t.Fatalf("embedded grammar parse failed")
	}
}
