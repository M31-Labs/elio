package graph_test

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
	"m31labs.dev/elio/stdlib"
)

func kernels(t *testing.T) []graph.Kernel {
	t.Helper()
	var ks []graph.Kernel
	ev, err := stdlib.OceanEvolve()
	must(t, err)
	as, err := stdlib.OceanAssemble()
	must(t, err)
	mp, err := stdlib.OceanMip()
	must(t, err)
	sp, err := stdlib.OceanSpectrum()
	must(t, err)
	fc, err := stdlib.FFT2DCompute(64)
	must(t, err)
	fg, err := stdlib.FFT2DGather(64)
	must(t, err)
	ks = append(ks,
		graph.Kernel{Name: "ocean_evolve", Entry: "ocean_evolve", Module: ev, Gather: true},
		graph.Kernel{Name: "ocean_assemble", Entry: "ocean_assemble", Module: as, Gather: true},
		graph.Kernel{Name: "ocean_mip", Entry: "ocean_mip", Module: mp, Gather: true},
		graph.Kernel{Name: "ocean_spectrum", Entry: "ocean_spectrum", Module: sp},
		graph.Kernel{Name: "fft_compute_64", Entry: "fft_rows", Module: fc},
		graph.Kernel{Name: "fft_gather_64", Entry: "fft_stage", Module: fg, Gather: true},
	)
	return ks
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func evolveUniform() graph.Uniform {
	return graph.Uniform{Name: "ev", Values: map[string]any{
		"n": 64, "_pad0": 0, "_pad1": 0, "_pad2": 0, "t": "param:t", "tile": 250.0, "loopPeriod": 1000.0, "_pad4": 0,
	}}
}

// validGraph is a small WebGPU graph: evolve, in-place row FFT, assemble, one mip.
func validGraph() graph.Graph {
	return graph.Graph{
		Name: "mini", Backend: graph.WebGPU, Tier: "high", Version: 1,
		Params: []graph.Param{
			{Name: "time", Type: graph.F32, Source: graph.Source{Kind: graph.SourceFrame, Name: "time"}},
			{Name: "loop", Type: graph.F32, Source: graph.Source{Kind: graph.SourceValue, Value: 1000}},
			{Name: "t", Type: graph.F32, Source: graph.Source{Kind: graph.SourceExpr, Op: "mod", Args: []any{"param:time", "param:loop"}}},
			{Name: "lambda", Type: graph.F32, Source: graph.Source{Kind: graph.SourceValue, Value: 1}, Tween: 2},
		},
		Resources: []graph.Resource{
			{Name: "h0", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096},
			{Name: "omega", Kind: "buffer", ElementType: "f32", Count: 4096},
			{Name: "A", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096},
			{Name: "B", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096},
			{Name: "disp", Kind: "texture", Format: "rgba16float", Width: 64, Height: 64, MipLevels: 2},
			{Name: "slope", Kind: "texture", Format: "rgba16float", Width: 64, Height: 64, MipLevels: 1},
		},
		Passes: []graph.Pass{
			{Name: "evolve", Type: "kernel", Kernel: "ocean_evolve", Entry: "ocean_evolve", When: "always",
				Dispatch: &[3]int{64, 1, 1},
				Bindings: []graph.Binding{{Name: "h0", Resource: "h0"}, {Name: "omega", Resource: "omega"}, {Name: "outA", Resource: "A"}, {Name: "outB", Resource: "B"}},
				Uniforms: []graph.Uniform{evolveUniform()}},
			{Name: "fft", Type: "kernel", Kernel: "fft_compute_64", Entry: "fft_rows", When: "changed:lambda",
				Dispatch: &[3]int{64, 1, 1},
				Bindings: []graph.Binding{{Name: "data", Resource: "A"}}},
			{Name: "assemble", Type: "kernel", Kernel: "ocean_assemble", Entry: "ocean_assemble", When: "always",
				Dispatch: &[3]int{64, 1, 1},
				Bindings: []graph.Binding{{Name: "fieldA", Resource: "A"}, {Name: "fieldB", Resource: "B"}, {Name: "disp", Resource: "disp"}, {Name: "slope", Resource: "slope"}},
				Uniforms: []graph.Uniform{{Name: "assem", Values: map[string]any{"n": 64, "_pad0": 0, "_pad1": 0, "_pad2": 0, "lambda": "param:lambda", "_pad3": 0, "_pad4": 0, "_pad5": 0}}}},
			{Name: "mip1", Type: "kernel", Kernel: "ocean_mip", Entry: "ocean_mip", When: "always",
				Dispatch: &[3]int{16, 1, 1},
				Bindings: []graph.Binding{{Name: "src", Resource: "disp", Level: 0}, {Name: "dst", Resource: "disp", Level: 1}},
				Uniforms: []graph.Uniform{{Name: "mip", Values: map[string]any{"srcLevel": 0, "dstWidth": 32, "dstHeight": 32, "pad": 0}}}},
		},
		FrameSwaps: [][2]string{{"A", "B"}},
		Exports:    map[string]string{"disp": "disp", "slope": "slope"},
	}
}

func TestValidGraphPasses(t *testing.T) {
	if err := graph.Validate(validGraph(), kernels(t)); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRejects(t *testing.T) {
	ks := kernels(t)
	cases := []struct {
		name   string
		mutate func(g *graph.Graph)
		want   string
	}{
		{"unknown binding name", func(g *graph.Graph) { g.Passes[0].Bindings[0].Name = "nope" }, "no binding"},
		{"binding missing", func(g *graph.Graph) { g.Passes[0].Bindings = g.Passes[0].Bindings[:3] }, `binding "outB" is not bound`},
		{"binding bound twice", func(g *graph.Graph) {
			g.Passes[0].Bindings = append(g.Passes[0].Bindings, g.Passes[0].Bindings[0])
		}, "bound more than once"},
		{"buffer bound to texture", func(g *graph.Graph) { g.Passes[0].Bindings[0].Resource = "disp" }, "needs a buffer"},
		{"element type mismatch", func(g *graph.Graph) { g.Passes[0].Bindings[0].Resource = "omega" }, "element type"},
		{"texture format mismatch", func(g *graph.Graph) { g.Resources[4].Format = "rgba32float" }, "format"},
		{"unknown resource", func(g *graph.Graph) { g.Passes[0].Bindings[0].Resource = "ghost" }, `unknown resource "ghost"`},
		{"level out of range", func(g *graph.Graph) { g.Passes[3].Bindings[1].Level = 5 }, "level 5"},
		{"reads what it writes", func(g *graph.Graph) { g.Passes[0].Bindings[0].Resource = "A" }, "reads and writes"},
		{"mip reads own level", func(g *graph.Graph) { g.Passes[3].Bindings[0].Level = 1 }, "reads and writes"},
		{"in place on webgl2", func(g *graph.Graph) { g.Backend = graph.WebGL2 }, "gather"},
		{"dispatch too small", func(g *graph.Graph) { g.Passes[0].Dispatch = &[3]int{32, 1, 1} }, "covers"},
		{"missing uniform field", func(g *graph.Graph) { delete(g.Passes[0].Uniforms[0].Values, "tile") }, `no value for field "tile"`},
		{"unknown uniform field", func(g *graph.Graph) { g.Passes[0].Uniforms[0].Values["zzz"] = 1 }, `no field "zzz"`},
		{"undeclared param", func(g *graph.Graph) { g.Passes[0].Uniforms[0].Values["t"] = "param:ghost" }, `undeclared param "ghost"`},
		{"param type mismatch", func(g *graph.Graph) { g.Passes[0].Uniforms[0].Values["n"] = "param:t" }, "type"},
		{"when undeclared", func(g *graph.Graph) { g.Passes[1].When = "changed:ghost" }, `undeclared param "ghost"`},
		{"expr references later param", func(g *graph.Graph) {
			g.Params[2].Source.Args = []any{"param:time", "param:late"}
			g.Params = append(g.Params, graph.Param{Name: "late", Type: graph.F32, Source: graph.Source{Kind: graph.SourceValue}})
		}, "earlier"},
		{"unknown frame source", func(g *graph.Graph) { g.Params[0].Source.Name = "wall" }, "frame source"},
		{"bad op", func(g *graph.Graph) { g.Params[2].Source.Op = "pow" }, "op"},
		{"tween on expr", func(g *graph.Graph) { g.Params[2].Tween = 1 }, "tween"},
		{"export unknown", func(g *graph.Graph) { g.Exports["x"] = "ghost" }, "export"},
		{"export a buffer on webgl2", func(g *graph.Graph) { g.Backend = graph.WebGL2; g.Exports["x"] = "A" }, "must be a texture"},
		{"swap size mismatch", func(g *graph.Graph) { g.FrameSwaps = [][2]string{{"A", "omega"}} }, "swap"},
		{"duplicate pass name", func(g *graph.Graph) { g.Passes[1].Name = "evolve" }, "duplicate"},
		{"unknown kernel", func(g *graph.Graph) { g.Passes[0].Kernel = "ghost" }, "unknown kernel"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := validGraph()
			// deep-enough copies of the pieces the cases edit
			g.Passes = append([]graph.Pass(nil), g.Passes...)
			for i := range g.Passes {
				g.Passes[i].Bindings = append([]graph.Binding(nil), g.Passes[i].Bindings...)
			}
			g.Passes[0].Uniforms = []graph.Uniform{evolveUniform()}
			g.Params = append([]graph.Param(nil), g.Params...)
			g.Params[2].Source.Args = append([]any(nil), g.Params[2].Source.Args...)
			g.Resources = append([]graph.Resource(nil), g.Resources...)
			g.Exports = map[string]string{"disp": "disp", "slope": "slope"}
			c.mutate(&g)
			err := graph.Validate(g, ks)
			if err == nil {
				t.Fatal("want an error, got none")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("error %q does not contain %q", err, c.want)
			}
		})
	}
}
