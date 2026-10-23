package graph_test

import (
	"encoding/json"
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

func exprParam(args ...any) graph.Param {
	return graph.Param{Name: "x", Type: graph.F32, Source: graph.Source{Kind: graph.SourceExpr, Op: "sub", Args: args}}
}

// An expression argument may itself be an expression, so a chain of single-use
// steps is one param.
func TestNestedExpressionAccepted(t *testing.T) {
	nested := exprParam("param:t", graph.E("mul", "param:loop", graph.E("floor", graph.E("div", "param:time", 5.0))))
	if err := graph.Validate(withParams(nested), kernels(t)); err != nil {
		t.Fatal(err)
	}
}

func TestNestedExpressionRejected(t *testing.T) {
	cases := []struct {
		name string
		p    graph.Param
		want string
	}{
		{"wrong arity", exprParam("param:t", graph.E("floor", 1.0, 2.0)), "takes 1 args"},
		{"unknown op", exprParam("param:t", graph.E("pow", 1.0, 2.0)), `unknown op "pow"`},
		{"later param", exprParam("param:t", graph.E("add", "param:late", 1.0)), "earlier"},
		{"bad argument", exprParam("param:t", graph.E("add", true, 1.0)), "not a number"},
	}
	for _, c := range cases {
		err := graph.Validate(withParams(c.p), kernels(t))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want %q", c.name, err, c.want)
		}
	}
}

func TestNestedExpressionJSONRoundTrip(t *testing.T) {
	g := withParams(exprParam("param:t", graph.E("mul", "param:loop", 2.0)))
	b, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `{"op":"mul","args":["param:loop",2]}`) {
		t.Errorf("nested expression JSON: %s", b)
	}
	var back graph.Graph
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	// Decoded, a nested expression is a map; it must validate and expand alike.
	if err := graph.Validate(back, kernels(t)); err != nil {
		t.Fatalf("round-tripped graph: %v", err)
	}
}
