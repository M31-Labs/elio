package graph_test

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

func TestBuilderBuildsAndFillsPadding(t *testing.T) {
	ks := kernels(t)
	b := graph.NewBuilder("mini", graph.WebGPU, "high", ks)
	b.Value("loop", graph.F32, 1000).Frame("time", "time")
	b.Expr("t", graph.F32, "mod", graph.Ref("time"), graph.Ref("loop"))
	b.Buffer("h0", "vec4<f32>", 4096, 0).Buffer("omega", "f32", 4096, 0)
	b.Buffer("A", "vec4<f32>", 4096, 0).Buffer("B", "vec4<f32>", 4096, 0)
	u := b.Uniform("ocean_evolve", "ev", map[string]any{"n": 64, "t": graph.Ref("t"), "tile": 250.0, "loopPeriod": 1000.0})
	if u.Values["_pad0"] != 0 || u.Values["_pad4"] != 0 {
		t.Fatalf("padding not filled: %v", u.Values)
	}
	b.Pass(graph.Pass{Name: "evolve", Type: "kernel", Kernel: "ocean_evolve", Entry: "ocean_evolve", When: "always",
		Dispatch: b.Workgroups("ocean_evolve", "ocean_evolve", 4096),
		Bindings: []graph.Binding{{Name: "h0", Resource: "h0"}, {Name: "omega", Resource: "omega"}, {Name: "outA", Resource: "A"}, {Name: "outB", Resource: "B"}},
		Uniforms: []graph.Uniform{u}})
	g, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	if g.Passes[0].Dispatch[0] != 64 {
		t.Errorf("dispatch %v", *g.Passes[0].Dispatch)
	}
}

func TestBuilderReportsFirstError(t *testing.T) {
	b := graph.NewBuilder("mini", graph.WebGPU, "high", kernels(t))
	b.Workgroups("ghost", "x", 1)
	if _, err := b.Build(); err == nil || !strings.Contains(err.Error(), `unknown kernel "ghost"`) {
		t.Fatalf("err = %v", err)
	}
	b = graph.NewBuilder("mini", graph.WebGPU, "high", kernels(t))
	b.Expr("x", graph.F32, "add", graph.Ref("later"), 1.0)
	if _, err := b.Build(); err == nil || !strings.Contains(err.Error(), "earlier") {
		t.Fatalf("err = %v", err)
	}
}
