package parse

import (
	"os"
	"testing"

	"m31labs.dev/elio/ir"
)

var benchmarkModule *ir.Module

func BenchmarkParseCull(b *testing.B) {
	data, err := os.ReadFile("testdata/cull.elio")
	if err != nil {
		b.Fatal(err)
	}
	src := string(data)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkModule, err = Parse(src)
		if err != nil {
			b.Fatal(err)
		}
	}
}
