package graph_test

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

// copyGraph is a WebGL2 graph whose pass is 128 texels wide: it reads S1, a
// 128-wide view of the 64-wide buffer S0 made by a copy pass, and writes D.
func copyGraph() graph.Graph {
	return graph.Graph{
		Name: "copy", Backend: graph.WebGL2, Tier: "high", Version: 1, Params: []graph.Param{},
		Resources: []graph.Resource{
			{Name: "S0", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096, Width: 64},
			{Name: "S1", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096, Width: 128},
			{Name: "D", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096, Width: 128},
		},
		Passes: []graph.Pass{
			{Name: "widen", Type: "copy", From: "S0", To: "S1", Width: 128},
			{Name: "stage", Type: "kernel", Kernel: "fft_gather_64", Entry: "fft_stage",
				Target:   &graph.Target{Width: 128, Height: 32, ElioWidth: 128},
				Bindings: []graph.Binding{{Name: "src", Resource: "S1"}, {Name: "dst", Resource: "D"}},
				Uniforms: []graph.Uniform{{Name: "fft", Values: map[string]any{"n": 64, "stage": 0, "axis": 0}}}},
		},
		FrameSwaps: [][2]string{}, Exports: map[string]string{},
	}
}

func TestCopyPassWidensABuffer(t *testing.T) {
	if err := graph.Validate(copyGraph(), kernels(t)); err != nil {
		t.Fatal(err)
	}
	// A gather pass reads each buffer at the buffer's own width, so it may read S0 (64 wide) in a 128-wide pass without a copy.
	g := copyGraph()
	g.Passes[1].Bindings[0].Resource = "S0"
	g.Passes = g.Passes[1:]
	if err := graph.Validate(g, kernels(t)); err != nil {
		t.Fatalf("a read at the buffer's own width is valid: %v", err)
	}
}

func TestCopyPassRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(g *graph.Graph)
		want   string
	}{
		{"width differs from the view", func(g *graph.Graph) { g.Passes[0].Width = 256 }, "width"},
		{"unknown source", func(g *graph.Graph) { g.Passes[0].From = "ghost" }, "copy"},
		{"count differs", func(g *graph.Graph) { g.Resources[1].Count = 8192 }, "copy"},
		{"element type differs", func(g *graph.Graph) { g.Resources[1].ElementType = "f32" }, "copy"},
		{"texture source", func(g *graph.Graph) {
			g.Resources[0] = graph.Resource{Name: "S0", Kind: "texture", Format: "rgba16float", Width: 64, Height: 64, MipLevels: 1}
		}, "copy"},
		{"webgpu", func(g *graph.Graph) { g.Backend = graph.WebGPU }, "webgl2"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := copyGraph()
			g.Passes = append([]graph.Pass(nil), g.Passes...)
			g.Resources = append([]graph.Resource(nil), g.Resources...)
			c.mutate(&g)
			err := graph.Validate(g, kernels(t))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want %q", err, c.want)
			}
		})
	}
}

// A copy inside a repeat follows the repeat's swaps like any other pass.
func TestCopyPassFollowsSwaps(t *testing.T) {
	g := copyGraph()
	g.Resources = append(g.Resources, graph.Resource{Name: "S0b", Kind: "buffer", ElementType: "vec4<f32>", Count: 4096, Width: 64})
	body := g.Passes[0]
	g.Passes = []graph.Pass{{Name: "twice", Type: "repeat", Count: 2, Index: "i", Swap: [][2]string{{"S0", "S0b"}}, Body: []graph.Pass{body}}, g.Passes[1]}
	flat, err := graph.Expand(g)
	if err != nil {
		t.Fatal(err)
	}
	if flat.Passes[0].From != "S0" || flat.Passes[1].From != "S0b" || flat.Passes[0].Name != "twice[0].widen" {
		t.Errorf("expanded copies read %s then %s", flat.Passes[0].From, flat.Passes[1].From)
	}
}
