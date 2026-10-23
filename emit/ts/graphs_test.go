package ts

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

func miniGraph() graph.Graph {
	return graph.Graph{
		Name: "mini", Backend: graph.WebGL2, Tier: "high", Version: 1,
		Params: []graph.Param{{Name: "k", Type: graph.F32, Source: graph.Source{Kind: graph.SourceValue, Value: 2}, Tween: 1}},
		Resources: []graph.Resource{
			{Name: "src", Kind: "buffer", ElementType: "vec4<f32>", Count: 16, Width: 4},
			{Name: "dst", Kind: "texture", Format: "rgba16float", Width: 4, Height: 4, MipLevels: 3},
		},
		Passes: []graph.Pass{
			{Name: "mini", Type: "kernel", Kernel: "mini_kernel", Entry: "mini", When: "always",
				Target:   &graph.Target{Width: 4, Height: 4, ElioWidth: 4},
				Bindings: []graph.Binding{{Name: "src", Resource: "src"}, {Name: "dst", Resource: "dst"}},
				Uniforms: []graph.Uniform{{Name: "p", Values: map[string]any{"n": 4, "k": "param:k"}}}},
			{Name: "mips", Type: "mipmap", Resource: "dst", When: "always"},
		},
		FrameSwaps: [][2]string{},
		Exports:    map[string]string{"out": "dst"},
	}
}

func TestEmitGraphs(t *testing.T) {
	set := miniSet(t, true)
	set.Graphs = []NamedGraph{{Key: "webgl2/high", Graph: miniGraph()}}
	global, err := Emit(set, Meta{Version: "v", Global: "G"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"    graphs: {\n", `"webgl2/high": {`, `"kind":"value"`, `"elioWidth":4`, `"when":"always"`} {
		if !strings.Contains(global, want) {
			t.Errorf("global output lacks %q", want)
		}
	}
	if strings.Index(global, "graphs: {") < strings.Index(global, "kernels: {") {
		t.Error("graphs should follow kernels")
	}
	es, err := Emit(set, Meta{Version: "v"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(es, "export const graphs") || !strings.Contains(es, `"webgl2/high"`) {
		t.Error("ES output lacks graphs")
	}
	again, _ := Emit(set, Meta{Version: "v", Global: "G"})
	if again != global {
		t.Error("graph output is not deterministic")
	}
	plain, _ := Emit(miniSet(t, true), Meta{Version: "v", Global: "G"})
	if strings.Contains(plain, "graphs:") {
		t.Error("a set without graphs emitted graphs")
	}
}

func TestEmitRejectsInvalidGraph(t *testing.T) {
	set := miniSet(t, true)
	g := miniGraph()
	g.Passes[0].Bindings = g.Passes[0].Bindings[:1]
	set.Graphs = []NamedGraph{{Key: "webgl2/high", Graph: g}}
	if _, err := Emit(set, Meta{Version: "v"}); err == nil || !strings.Contains(err.Error(), `binding "dst" is not bound`) {
		t.Fatalf("err = %v", err)
	}
	set.Graphs = []NamedGraph{{Key: "wrong", Graph: miniGraph()}}
	if _, err := Emit(set, Meta{Version: "v"}); err == nil || !strings.Contains(err.Error(), "key") {
		t.Fatalf("err = %v", err)
	}
}

func TestEmitSharesIdenticalParams(t *testing.T) {
	set := miniSet(t, true)
	a, b, c := miniGraph(), miniGraph(), miniGraph()
	b.Tier, c.Tier = "medium", "low"
	c.Params = append([]graph.Param(nil), c.Params...)
	c.Params[0].Source.Value = 9
	set.Graphs = []NamedGraph{{Key: "webgl2/high", Graph: a}, {Key: "webgl2/medium", Graph: b}, {Key: "webgl2/low", Graph: c}}
	out, err := Emit(set, Meta{Version: "v", Global: "G", Backend: "webgl2"})
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(out, `"paramsFrom":"webgl2/high"`); n != 1 {
		t.Errorf("%d graphs share the first graph's params, want 1 (medium)", n)
	}
	if n := strings.Count(out, `"params":`); n != 2 {
		t.Errorf("%d graphs carry their own params, want 2 (high and low)", n)
	}
}
