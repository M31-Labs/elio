// Command elio-grammar regenerates the embedded parser blob used by parse.
package main

import (
	"fmt"
	"os"

	"github.com/odvcencio/gotreesitter/grammargen"
	"m31labs.dev/elio/grammar"
)

func main() {
	_, blob, err := grammargen.GenerateLanguageAndBlob(grammar.ElioGrammar())
	if err != nil {
		fmt.Fprintln(os.Stderr, "elio-grammar: generate parse table:", err)
		os.Exit(1)
	}
	if err := os.WriteFile("grammar.bin", blob, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "elio-grammar: write grammar.bin:", err)
		os.Exit(1)
	}
	fmt.Printf("regenerated grammar.bin (%d bytes)\n", len(blob))
}
