package ts

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

func miniGPUGraph() graph.Graph {
	g := miniGraph()
	g.Backend = graph.WebGPU
	g.Resources[0].Width = 0
	g.Passes = []graph.Pass{{Name: "mini", Type: "kernel", Kernel: "mini_kernel", Entry: "mini", When: "always",
		Dispatch: &[3]int{1, 1, 1},
		Bindings: g.Passes[0].Bindings,
		Uniforms: g.Passes[0].Uniforms}}
	return g
}

func bothGraphs() []NamedGraph {
	return []NamedGraph{{Key: "webgpu/high", Graph: miniGPUGraph()}, {Key: "webgl2/high", Graph: miniGraph()}}
}

func TestEmitBackendWebGPU(t *testing.T) {
	set := miniSet(t, true)
	set.Graphs = bothGraphs()
	out, err := Emit(set, Meta{Version: "v", Global: "G", Backend: "webgpu"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"\n      wgsl:", "\n      workgroup:", `"webgpu/high"`} {
		if !strings.Contains(out, want) {
			t.Errorf("webgpu module lacks %q", want)
		}
	}
	for _, bad := range []string{"\n      glsl:", "gatherLayout", "uniformsStd140", `"webgl2/high"`} {
		if strings.Contains(out, bad) {
			t.Errorf("webgpu module has %q", bad)
		}
	}
}

func TestEmitBackendWebGL2(t *testing.T) {
	set := miniSet(t, true)
	set.Graphs = bothGraphs()
	out, err := Emit(set, Meta{Version: "v", Global: "G", Backend: "webgl2"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"\n      glsl:", "gatherLayout", "uniforms:", `"webgl2/high"`} {
		if !strings.Contains(out, want) {
			t.Errorf("webgl2 module lacks %q", want)
		}
	}
	for _, bad := range []string{"\n      wgsl:", "\n      workgroup:", "uniformsStd140", `"webgpu/high"`} {
		if strings.Contains(out, bad) {
			t.Errorf("webgl2 module has %q", bad)
		}
	}
	// The webgl2 uniforms use std140 offsets (the same here: n at 0, k at 4).
	// A kernel that is not gather-compatible is left out.
	plain := miniSet(t, false)
	plain.Graphs = []NamedGraph{{Key: "webgpu/high", Graph: miniGPUGraph()}}
	out, err = Emit(plain, Meta{Version: "v", Global: "G", Backend: "webgl2"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "miniKernel") {
		t.Error("non-gather kernel in the webgl2 module")
	}
}

func TestEmitBackendES(t *testing.T) {
	set := miniSet(t, true)
	set.Graphs = bothGraphs()
	out, err := Emit(set, Meta{Version: "v", Backend: "webgl2"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "WGSL") || !strings.Contains(out, "miniKernelGLSL") {
		t.Error("ES webgl2 module has the wrong shaders")
	}
}

func TestEmitBackendUnknown(t *testing.T) {
	if _, err := Emit(miniSet(t, true), Meta{Version: "v", Backend: "metal"}); err == nil {
		t.Fatal("want an error for an unknown backend")
	}
}

func TestEmitHashCoversGraphs(t *testing.T) {
	set := miniSet(t, true)
	plain, _ := Emit(set, Meta{Version: "v"})
	set.Graphs = []NamedGraph{{Key: "webgl2/high", Graph: miniGraph()}}
	with, _ := Emit(set, Meta{Version: "v"})
	g := miniGraph()
	g.Params[0].Source.Value = 3
	set.Graphs = []NamedGraph{{Key: "webgl2/high", Graph: g}}
	changed, _ := Emit(set, Meta{Version: "v"})
	sum := func(s string) string {
		i := strings.Index(s, "// sha256: ")
		return s[i : i+80]
	}
	if sum(plain) == sum(with) || sum(with) == sum(changed) {
		t.Errorf("hash ignores graphs:\n%s\n%s\n%s", sum(plain), sum(with), sum(changed))
	}
}
