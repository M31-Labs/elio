package run

import (
	"testing"

	"m31labs.dev/elio/ir"
)

var benchmarkCullDrawArgs []float64

func BenchmarkRunCull256(b *testing.B) {
	mod := ir.CullKernel()
	const n = 256
	planes := []any{
		[]float64{1, 0, 0, 10},
		[]float64{-1, 0, 0, 10},
		[]float64{0, 1, 0, 10},
		[]float64{0, -1, 0, 10},
		[]float64{0, 0, 1, 10},
		[]float64{0, 0, -1, 10},
	}
	in := make([]any, n)
	for i := range in {
		x := float64(i%32) - 16
		in[i] = record(x, 0, 0)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out := make([]any, n)
		drawArgs := []float64{36, 0, 0, 0}
		mem := &Memory{Vars: map[string]any{
			"cull": map[string]any{
				"planes":      planes,
				"vertexCount": float64(36),
				"radius":      float64(0),
			},
			"input":    in,
			"output":   out,
			"drawArgs": drawArgs,
		}}
		if err := Run(mod, "main", n, mem); err != nil {
			b.Fatal(err)
		}
		benchmarkCullDrawArgs = drawArgs
	}
}
