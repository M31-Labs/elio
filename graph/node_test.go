package graph_test

import (
	"encoding/json"
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

func withParams(ps ...graph.Param) graph.Graph {
	g := validGraph()
	g.Params = append(append([]graph.Param(nil), g.Params...), ps...)
	return g
}

func TestNodeSourceAccepted(t *testing.T) {
	for _, f := range []string{"x", "z", "heading", "speed"} {
		g := withParams(graph.Param{Name: "hull_" + f, Type: graph.F32, Source: graph.Source{Kind: graph.SourceNode, Node: "ship", Field: f}})
		if err := graph.Validate(g, kernels(t)); err != nil {
			t.Errorf("field %s: %v", f, err)
		}
	}
}

func TestNodeSourceRejected(t *testing.T) {
	cases := []struct {
		name string
		p    graph.Param
		want string
	}{
		{"unknown field", graph.Param{Name: "a", Type: graph.F32, Source: graph.Source{Kind: graph.SourceNode, Node: "ship", Field: "y"}}, "node field"},
		{"no node", graph.Param{Name: "a", Type: graph.F32, Source: graph.Source{Kind: graph.SourceNode, Field: "x"}}, "node id"},
		{"not f32", graph.Param{Name: "a", Type: graph.U32, Source: graph.Source{Kind: graph.SourceNode, Node: "ship", Field: "x"}}, "f32"},
	}
	for _, c := range cases {
		err := graph.Validate(withParams(c.p), kernels(t))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want %q", c.name, err, c.want)
		}
	}
}

func TestNodeSourceJSON(t *testing.T) {
	b, err := json.Marshal(graph.Param{Name: "hullX", Type: graph.F32, Source: graph.Source{Kind: graph.SourceNode, Node: "ship", Field: "x"}})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"name":"hullX","type":"f32","source":{"field":"x","kind":"node","node":"ship"}}`
	if string(b) != want {
		t.Errorf("json %s, want %s", b, want)
	}
}

// A bool param reads as 1 or 0 in an expression, a delta or an accumulate, so a
// graph can turn a switch into a gain or detect the frame it flipped.
func TestBoolParamUsableAsNumber(t *testing.T) {
	g := withParams(
		graph.Param{Name: "on", Type: graph.Bool, Source: graph.Source{Kind: graph.SourceValue, Value: 1}},
		graph.Param{Name: "gain", Type: graph.F32, Source: graph.Source{Kind: graph.SourceExpr, Op: "mul", Args: []any{"param:on", 2.0}}},
		graph.Param{Name: "edge", Type: graph.I32, Source: graph.Source{Kind: graph.SourceDelta, Of: "param:on"}},
	)
	if err := graph.Validate(g, kernels(t)); err != nil {
		t.Fatal(err)
	}
}
