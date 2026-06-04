package parse

import _ "embed"

//go:generate go run ../cmd/elio-grammar

// grammarBlob is the pre-generated parse table for grammar.ElioGrammar().
// TestGrammarBinIsCurrent guards it against drift.
//
//go:embed grammar.bin
var grammarBlob []byte
