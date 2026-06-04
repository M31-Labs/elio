package metal

import (
	"testing"

	"m31labs.dev/elio/ir"
)

var benchmarkMetalSource string

func BenchmarkEmitCull(b *testing.B) {
	mod := ir.CullKernel()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		src, err := Emit(mod)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkMetalSource = src
	}
	b.ReportMetric(float64(len(benchmarkMetalSource)), "source_bytes")
}
