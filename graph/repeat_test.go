package graph_test

import (
	"encoding/json"
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

// repeatGraph is a WebGL2 inverse FFT of one 64x64 buffer: two repeats of six
// stages (rows then columns), swapping A and A2 after each stage.
func repeatGraph(count int) graph.Graph {
	stage := func(axis int) graph.Pass {
		return graph.Pass{Name: "axis", Type: "repeat", Count: count, Index: "s", Swap: [][2]string{{"A", "A2"}}, When: "always",
			Body: []graph.Pass{{Name: "stage", Type: "kernel", Kernel: "fft_gather_64", Entry: "fft_stage", When: "always",
				Target:   &graph.Target{Width: 64, Height: 64, ElioWidth: 64},
				Bindings: []graph.Binding{{Name: "src", Resource: "A"}, {Name: "dst", Resource: "A2"}},
				Uniforms: []graph.Uniform{{Name: "fft", Values: map[string]any{"n": 64, "stage": "param:s", "axis": axis}}}}}}
	}
	r0, r1 := stage(0), stage(1)
	r0.Name, r1.Name = "rows", "cols"
	return graph.Graph{
		Name: "fftonly", Backend: graph.WebGL2, Tier: "high", Version: 1,
		Params: []graph.Param{},
		Resources: []graph.Resource{
			{Name: "A", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096, Width: 64},
			{Name: "A2", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096, Width: 64},
		},
		Passes:     []graph.Pass{r0, r1},
		FrameSwaps: [][2]string{},
		Exports:    map[string]string{},
	}
}

func TestRepeatExpands(t *testing.T) {
	g := repeatGraph(6)
	flat, err := graph.Expand(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(flat.Passes) != 12 {
		t.Fatalf("%d passes", len(flat.Passes))
	}
	for i, p := range flat.Passes {
		src, dst := "A", "A2"
		if i%2 == 1 {
			src, dst = "A2", "A"
		}
		if p.Type != "kernel" || p.Bindings[0].Resource != src || p.Bindings[1].Resource != dst {
			t.Errorf("pass %d (%s): binds %v", i, p.Name, p.Bindings)
		}
		axis, stage := i/6, i%6
		v := p.Uniforms[0].Values
		if v["axis"] != axis || v["stage"] != stage {
			t.Errorf("pass %d: axis %v stage %v, want %d %d", i, v["axis"], v["stage"], axis, stage)
		}
	}
	if err := graph.Validate(g, kernels(t)); err != nil {
		t.Fatal(err)
	}
	// The source graph is untouched by expansion.
	if g.Passes[0].Body[0].Uniforms[0].Values["stage"] != "param:s" {
		t.Error("Expand changed its input")
	}
}

func TestRepeatRejects(t *testing.T) {
	ks := kernels(t)
	cases := []struct {
		name   string
		mutate func(g *graph.Graph)
		want   string
	}{
		{"odd swaps never restored", func(g *graph.Graph) { g.Passes = g.Passes[:1]; g.Passes[0].Count = 5 }, "not restored"},
		{"zero count", func(g *graph.Graph) { g.Passes[0].Count = 0 }, "count"},
		{"no index", func(g *graph.Graph) { g.Passes[0].Index = "" }, "index"},
		{"index shadows param", func(g *graph.Graph) {
			g.Params = append(g.Params, graph.Param{Name: "s", Type: graph.F32, Source: graph.Source{Kind: graph.SourceValue}})
		}, "index"},
		{"empty body", func(g *graph.Graph) { g.Passes[0].Body = nil }, "body"},
		{"swap of unknown resource", func(g *graph.Graph) { g.Passes[0].Swap = [][2]string{{"A", "ghost"}} }, "swap"},
		{"no swap reads and writes one buffer", func(g *graph.Graph) { g.Passes[0].Swap = nil; g.Passes[0].Body[0].Bindings[1].Resource = "A" }, "reads and writes"},
		{"unknown index in body", func(g *graph.Graph) { g.Passes[0].Body[0].Uniforms[0].Values["stage"] = "param:t" }, `undeclared param "t"`},
		{"index only inside body", func(g *graph.Graph) { g.Passes[1].Body[0].Uniforms[0].Values["stage"] = "param:s" }, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := repeatGraph(6)
			// Deep copy what the cases edit.
			g.Passes = append([]graph.Pass(nil), g.Passes...)
			for i := range g.Passes {
				body := append([]graph.Pass(nil), g.Passes[i].Body...)
				for j := range body {
					body[j].Bindings = append([]graph.Binding(nil), body[j].Bindings...)
					body[j].Uniforms = []graph.Uniform{{Name: "fft", Values: map[string]any{"n": 64, "stage": "param:s", "axis": 0}}}
				}
				g.Passes[i].Body = body
			}
			c.mutate(&g)
			err := graph.Validate(g, ks)
			if c.want == "" { // the index is valid in every repeat's own body
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("error %v does not contain %q", err, c.want)
			}
		})
	}
}

func TestPadFieldsMayBeOmitted(t *testing.T) {
	g := validGraph()
	g.Passes[0].Uniforms = []graph.Uniform{{Name: "ev", Values: map[string]any{"n": 64, "t": "param:t", "tile": 250.0, "loopPeriod": 1000.0}}}
	g.Passes[2].Uniforms = []graph.Uniform{{Name: "assem", Values: map[string]any{"n": 64, "lambda": "param:lambda"}}}
	if err := graph.Validate(g, kernels(t)); err != nil {
		t.Fatal(err)
	}
	// A field that is not padding still needs a value (mip's "pad" does).
	g.Passes[3].Uniforms = []graph.Uniform{{Name: "mip", Values: map[string]any{"srcLevel": 0, "dstWidth": 32, "dstHeight": 32}}}
	if err := graph.Validate(g, kernels(t)); err == nil || !strings.Contains(err.Error(), `field "pad"`) {
		t.Fatalf("err = %v", err)
	}
}

// mipRepeatGraph builds the WebGPU mip chain of one 64x64 texture as one repeat.
func mipRepeatGraph() graph.Graph {
	g := graph.Graph{
		Name: "mips", Backend: graph.WebGPU, Tier: "high", Version: 1, Params: []graph.Param{},
		Resources:  []graph.Resource{{Name: "tex", Kind: "texture", Format: "rgba16float", Width: 64, Height: 64, MipLevels: 7}},
		FrameSwaps: [][2]string{}, Exports: map[string]string{"tex": "tex"},
	}
	g.Passes = []graph.Pass{{Name: "mips", Type: "repeat", Count: 6, Index: "l", When: "always", Body: []graph.Pass{
		{Name: "mip", Type: "kernel", Kernel: "ocean_mip", Entry: "ocean_mip", When: "always",
			Dispatch: &[3]int{16, 1, 1}, // sized for the largest level; smaller levels exit early
			Bindings: []graph.Binding{
				{Name: "src", Resource: "tex", LevelAt: &graph.IndexExpr{Index: "l"}},
				{Name: "dst", Resource: "tex", LevelAt: &graph.IndexExpr{Index: "l", Offset: 1}}},
			Uniforms: []graph.Uniform{{Name: "mip", Values: map[string]any{
				"srcLevel": 0, "pad": 0,
				"dstWidth":  graph.IndexExpr{Index: "l", Offset: 1, Shr: 64},
				"dstHeight": graph.IndexExpr{Index: "l", Offset: 1, Shr: 64}}}}}}}}
	return g
}

func TestRepeatIndexExpressions(t *testing.T) {
	g := mipRepeatGraph()
	flat, err := graph.Expand(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(flat.Passes) != 6 {
		t.Fatalf("%d passes", len(flat.Passes))
	}
	wantW := []int{32, 16, 8, 4, 2, 1}
	for i, p := range flat.Passes {
		src, dst := p.Bindings[0].Level, p.Bindings[1].Level
		v := p.Uniforms[0].Values
		if src != i || dst != i+1 || v["dstWidth"] != wantW[i] || v["dstHeight"] != wantW[i] || p.Bindings[0].LevelAt != nil {
			t.Errorf("pass %d: levels %d->%d, dst %v x %v", i, src, dst, v["dstWidth"], v["dstHeight"])
		}
	}
	if err := graph.Validate(g, kernels(t)); err != nil {
		t.Fatal(err)
	}
	// A dispatch smaller than the first level fails the coverage rule.
	g = mipRepeatGraph()
	g.Passes[0].Body[0].Dispatch = &[3]int{4, 1, 1}
	if err := graph.Validate(g, kernels(t)); err == nil || !strings.Contains(err.Error(), "covers") {
		t.Fatalf("err = %v", err)
	}
	// An index that is not in scope fails.
	g = mipRepeatGraph()
	g.Passes[0].Body[0].Bindings[0].LevelAt = &graph.IndexExpr{Index: "z"}
	if err := graph.Validate(g, kernels(t)); err == nil || !strings.Contains(err.Error(), `index "z"`) {
		t.Fatalf("err = %v", err)
	}
}

// An empty when means "always" and a zero level is omitted, which keeps the
// emitted JSON short.
func TestDefaultsAreOmittedFromJSON(t *testing.T) {
	g := validGraph()
	for i := range g.Passes {
		g.Passes[i].When = ""
		for j := range g.Passes[i].Bindings {
			g.Passes[i].Bindings[j].Level = 0
		}
	}
	g.Passes[3].Bindings[1].Level = 1
	if err := graph.Validate(g, kernels(t)); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(g.Passes[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), `"when"`) || strings.Contains(string(b), `"level"`) {
		t.Errorf("defaults in JSON: %s", b)
	}
	flat, _ := graph.Expand(mipRepeatGraph())
	if len(flat.Passes) == 0 {
		t.Fatal("no passes")
	}
}
