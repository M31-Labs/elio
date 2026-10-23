package graph_test

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

// Tiers of one backend usually declare the same params. An emitted graph may
// name another graph to take its params and exported params from.
func TestShareAndUnshare(t *testing.T) {
	base := validGraph()
	base.ExportParams = []string{"t"}
	other := validGraph()
	other.Tier = "low"
	other.ExportParams = []string{"t"}
	if !graph.SameParams(base, other) {
		t.Fatal("identical params reported different")
	}
	other.Params = append([]graph.Param(nil), other.Params...)
	other.Params[1].Source.Value = 5
	if graph.SameParams(base, other) {
		t.Fatal("different params reported the same")
	}

	same := validGraph()
	same.Tier = "medium"
	same.ExportParams = []string{"t"}
	shared := graph.Share(same, "webgpu/high")
	if shared.ParamsFrom != "webgpu/high" || shared.Params != nil || shared.ExportParams != nil {
		t.Fatalf("Share: %+v", shared)
	}
	if len(same.Params) == 0 {
		t.Error("Share changed its input")
	}
	back := graph.Unshare(shared, base)
	if back.ParamsFrom != "" || !graph.SameParams(back, same) {
		t.Errorf("Unshare did not restore the params: %+v", back)
	}
	if err := graph.Validate(back, kernels(t)); err != nil {
		t.Fatal(err)
	}
	// A graph that still has a reference is not valid until it is resolved.
	if err := graph.Validate(shared, kernels(t)); err == nil || !strings.Contains(err.Error(), "paramsFrom") {
		t.Errorf("err = %v", err)
	}
}
