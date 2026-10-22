package kernelsets

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

var graphKeys = []string{"webgpu/high", "webgpu/medium", "webgpu/low", "webgl2/high", "webgl2/medium", "webgl2/low"}

func oceanGraphs(t *testing.T) (map[string]graph.Graph, []graph.Kernel) {
	t.Helper()
	gs, err := OceanGraphs()
	if err != nil {
		t.Fatal(err)
	}
	set, err := Ocean()
	if err != nil {
		t.Fatal(err)
	}
	return gs, GraphKernels(set)
}

func TestOceanGraphsValidate(t *testing.T) {
	gs, ks := oceanGraphs(t)
	if len(gs) != len(graphKeys) {
		t.Fatalf("%d graphs, want %d", len(gs), len(graphKeys))
	}
	for _, key := range graphKeys {
		g, ok := gs[key]
		if !ok {
			t.Fatalf("missing graph %s", key)
		}
		if key != g.Backend+"/"+g.Tier || g.Name != "ocean" || g.Version != 1 {
			t.Errorf("%s: header %s %s/%s v%d", key, g.Name, g.Backend, g.Tier, g.Version)
		}
		if err := graph.Validate(g, ks); err != nil {
			t.Errorf("%s: %v", key, err)
		}
	}
}

// Pass counts, derived by hand: per cascade, webgpu has 1 spectrum, 1 evolve,
// 4 FFT dispatches (rows and columns of A and B), 1 assemble and 2 (levels-1)
// mips; webgl2 replaces the FFT with 4 log2(n) stages and the mips with 2
// mipmap passes. Then foam (1 + mips: 10 on webgpu, 1 mipmap on webgl2) and,
// on webgpu, one probe.
func TestOceanGraphPassCounts(t *testing.T) {
	gs, _ := oceanGraphs(t)
	want := map[string]int{
		"webgpu/high":   4 + 4 + 16 + 4 + (12 + 3*16) + 1 + 10 + 1,
		"webgpu/medium": 4 + 4 + 16 + 4 + (12 + 3*14) + 1 + 10 + 1,
		"webgpu/low":    3 + 3 + 12 + 3 + 3*12 + 1 + 10 + 1,
		"webgl2/high":   4 + 4 + (24 + 3*32) + 4 + 8 + 1 + 1,
		"webgl2/medium": 4 + 4 + (24 + 3*28) + 4 + 8 + 1 + 1,
		"webgl2/low":    3 + 3 + 3*24 + 3 + 6 + 1 + 1,
	}
	for key, n := range want {
		if got := len(gs[key].Passes); got != n {
			t.Errorf("%s: %d passes, want %d", key, got, n)
		}
	}
}

func TestOceanGraphExportsAndParams(t *testing.T) {
	gs, _ := oceanGraphs(t)
	for _, key := range graphKeys {
		g := gs[key]
		cascades := 4
		if g.Tier == "low" {
			cascades = 3
		}
		want := []string{"foam"}
		for c := 0; c < cascades; c++ {
			want = append(want, "disp"+string(rune('0'+c)), "slope"+string(rune('0'+c)))
		}
		if g.Backend == "webgpu" {
			want = append(want, "probe")
		}
		if len(g.Exports) != len(want) {
			t.Errorf("%s: exports %v, want %v", key, g.Exports, want)
		}
		for _, n := range want {
			if _, ok := g.Exports[n]; !ok {
				t.Errorf("%s: no export %s", key, n)
			}
		}
		params := map[string]bool{}
		for _, p := range g.Params {
			params[p.Name] = true
		}
		for _, n := range []string{"dt", "cameraX", "cameraZ", "timeScale", "driftX", "driftZ", "windSpeed", "fetch", "gamma", "spread",
			"swell", "windDirDeg", "swellDirDeg", "seed", "loopPeriod", "lambda", "foamTau", "foamThreshold", "foamGain",
			"windDir", "swellDir", "simTime", "phaseTime", "waterX", "waterZ", "foamOriginX", "foamOriginZ", "foamShiftX", "foamShiftZ", "probeCount"} {
			if !params[n] {
				t.Errorf("%s: no param %s", key, n)
			}
		}
		// The spectrum passes run when a spectrum input changed, not every frame.
		for _, p := range g.Passes {
			if strings.Contains(p.Name, ".spectrum") && !strings.HasPrefix(p.When, "changed:") {
				t.Errorf("%s: %s runs %q", key, p.Name, p.When)
			}
		}
	}
}

func TestOceanGraphResolutions(t *testing.T) {
	gs, _ := oceanGraphs(t)
	size := func(g graph.Graph, name string) int {
		for _, r := range g.Resources {
			if r.Name == name {
				return r.Width
			}
		}
		return 0
	}
	for key, n := range map[string]int{"webgpu/high": 256, "webgpu/medium": 128, "webgpu/low": 64, "webgl2/high": 256, "webgl2/medium": 128, "webgl2/low": 64} {
		g := gs[key]
		if size(g, "dispTex0") != 64 || size(g, "dispTex1") != n || size(g, "dispTex2") != n || size(g, "foamTex") != 1024 {
			t.Errorf("%s: cascade sizes %d %d %d foam %d", key, size(g, "dispTex0"), size(g, "dispTex1"), size(g, "dispTex2"), size(g, "foamTex"))
		}
	}
}
