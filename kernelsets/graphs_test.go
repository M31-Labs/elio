package kernelsets

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

var graphKeys = []string{"webgpu/high", "webgpu/medium", "webgpu/low", "webgl2/high", "webgl2/medium", "webgl2/low"}

func loadGraphs(t *testing.T) (map[string]graph.Graph, []graph.Kernel) {
	t.Helper()
	gs, err := OceanGraphs()
	if err != nil {
		t.Fatal(err)
	}
	set, err := Ocean()
	if err != nil {
		t.Fatal(err)
	}
	return gs, set.GraphKernels()
}

func TestOceanGraphsValidate(t *testing.T) {
	gs, ks := loadGraphs(t)
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
// WebGL2 adds one copy pass per cascade and one for the wake state: the foam pass
// is 1,024 wide and reads each of those buffers at that width.
// The wake adds, per graph: source, the forward FFT (rows and columns), the
// symbol, the second FFT, the step, and the mip chain of the 256 x 256 wake
// texture (8 levels above the base on webgpu, one mipmap on webgl2). The FFT is
// 2 dispatches on webgpu and 8 stages per axis (16 passes) on webgl2.
const (
	wakeGPUPasses = 1 + 2 + 1 + 2 + 1 + 8
	wakeGL2Passes = 1 + 16 + 1 + 16 + 1 + 1
)

func TestOceanGraphPassCounts(t *testing.T) {
	gs, _ := loadGraphs(t)
	want := map[string]int{
		"webgpu/high":   4 + 4 + 16 + 4 + (12 + 3*16) + 1 + 10 + 1 + wakeGPUPasses,
		"webgpu/medium": 4 + 4 + 16 + 4 + (12 + 3*14) + 1 + 10 + 1 + wakeGPUPasses,
		"webgpu/low":    3 + 3 + 12 + 3 + 3*12 + 1 + 10 + 1 + wakeGPUPasses,
		"webgl2/high":   4 + 4 + (24 + 3*32) + 4 + 8 + 1 + 1 + wakeGL2Passes,
		"webgl2/medium": 4 + 4 + (24 + 3*28) + 4 + 8 + 1 + 1 + wakeGL2Passes,
		"webgl2/low":    3 + 3 + 3*24 + 3 + 6 + 1 + 1 + wakeGL2Passes,
	}
	for key, n := range want {
		flat, err := graph.Expand(gs[key])
		if err != nil {
			t.Fatal(err)
		}
		if got := len(flat.Passes); got != n {
			t.Errorf("%s: %d expanded passes, want %d", key, got, n)
		}
	}
	// FFT stages (WebGL2) and mip levels (WebGPU) are repeats, so no graph
	// lists more than 45 passes.
	for _, key := range graphKeys {
		if top := len(gs[key].Passes); top > 55 {
			t.Errorf("%s: %d top-level passes (limit 55); stages and mip levels should be repeats", key, top)
		}
	}
}

func TestOceanGraphExportsAndParams(t *testing.T) {
	gs, _ := loadGraphs(t)
	for _, key := range graphKeys {
		g := gs[key]
		cascades := 4
		if g.Tier == "low" {
			cascades = 3
		}
		want := []string{"foam", "wake"}
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
			"windDir", "swellDir", "simTime", "phaseTime", "waterX", "waterZ", "foamOriginX", "foamOriginZ", "foamShiftX", "foamShiftZ", "probeCount",
			"wakeEnabled", "wakeCell", "hullX", "hullZ", "hullHeading", "hullSpeed", "hullLength", "hullBeam", "wakeOriginX", "wakeOriginZ", "wakeShiftX", "wakeShiftZ"} {
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
	gs, _ := loadGraphs(t)
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

func TestOceanWithGraphsOrder(t *testing.T) {
	set, err := OceanWithGraphs()
	if err != nil {
		t.Fatal(err)
	}
	for i, ng := range set.Graphs {
		if ng.Key != graphKeys[i] {
			t.Errorf("graph %d is %s, want %s", i, ng.Key, graphKeys[i])
		}
	}
	if len(set.Graphs) != len(graphKeys) {
		t.Errorf("%d graphs", len(set.Graphs))
	}
}

// The surface samples cascade c at (xz - water) / size, so the graph exports
// the water offset wrapped to each cascade's tile: wrapping in f64 keeps the
// f32 sample position precise on long voyages.
func TestOceanGraphWaterWrap(t *testing.T) {
	gs, _ := loadGraphs(t)
	sizes := []float64{1600, 250, 37, 5.3}
	for _, key := range graphKeys {
		g := gs[key]
		// Every tier declares all four cascades' params (low simply ignores the
		// fourth), so the tiers share one param list.
		cascades := 4
		params := map[string]graph.Param{}
		for _, p := range g.Params {
			params[p.Name] = p
		}
		var want []string
		for c := 0; c < cascades; c++ {
			for _, ax := range []string{"X", "Z"} {
				name := fmt.Sprintf("waterMod%s%d", ax, c)
				want = append(want, name)
				p, ok := params[name]
				if !ok {
					t.Errorf("%s: no param %s", key, name)
					continue
				}
				s := p.Source
				if s.Kind != graph.SourceExpr || s.Op != "mod" || len(s.Args) != 2 || s.Args[0] != "param:water"+ax || s.Args[1] != sizes[c] {
					t.Errorf("%s: %s source %+v", key, name, s)
				}
			}
		}
		// The wake's surface params follow the water offsets (TestOceanGraphWake).
		if len(g.ExportParams) < len(want) || strings.Join(g.ExportParams[:len(want)], ",") != strings.Join(want, ",") {
			t.Errorf("%s: exportParams %v, want %v first", key, g.ExportParams, want)
		}
	}
}

// Foam and probe never form a large coordinate in f32: the graph computes, in
// f64, each cascade's wrapped window or point origin.
func TestOceanGraphWrappedOrigins(t *testing.T) {
	gs, _ := loadGraphs(t)
	sizes := []float64{1600, 250, 37, 5.3}
	for _, key := range graphKeys {
		g := gs[key]
		cascades := 4
		if g.Tier == "low" {
			cascades = 3
		}
		params := map[string]graph.Param{}
		for _, p := range g.Params {
			params[p.Name] = p
		}
		wantExpr := func(name, op string, a, b any) {
			t.Helper()
			p, ok := params[name]
			if !ok {
				t.Errorf("%s: no param %s", key, name)
				return
			}
			if p.Source.Kind != graph.SourceExpr || p.Source.Op != op || !reflect.DeepEqual(p.Source.Args, []any{a, b}) {
				t.Errorf("%s: %s source %+v, want %s(%v, %v)", key, name, p.Source, op, a, b)
			}
		}
		uniform := func(pass, field string) any {
			flat, err := graph.Expand(g)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range flat.Passes {
				if p.Name == pass {
					return p.Uniforms[0].Values[field]
				}
			}
			t.Fatalf("%s: no pass %s", key, pass)
			return nil
		}
		for _, ax := range []string{"X", "Z"} {
			for c := 0; c < cascades; c++ {
				wantExpr(fmt.Sprintf("foamOriginMod%d%s", c, ax), "mod", graph.E("mul", "param:foamOrigin"+ax, 0.5), sizes[c])
				if got := uniform("foam", fmt.Sprintf("om%s%d", ax, c)); got != fmt.Sprintf("param:foamOriginMod%d%s", c, ax) {
					t.Errorf("%s: foam om%s%d = %v", key, ax, c, got)
				}
			}
			// An unused cascade slot repeats the last real cascade's offset.
			if got := uniform("foam", "om"+ax+"3"); got != fmt.Sprintf("param:foamOriginMod%d%s", cascades-1, ax) {
				t.Errorf("%s: foam om%s3 = %v", key, ax, got)
			}
			if g.Backend != "webgpu" {
				if _, ok := params["probeOrigin"+ax]; ok {
					t.Errorf("%s: probe params on webgl2", key)
				}
				continue
			}
			if p := params["probeOrigin"+ax]; p.Source.Kind != graph.SourceValue || p.Type != graph.F32 {
				t.Errorf("%s: probeOrigin%s = %+v", key, ax, p)
			}
			for c := 0; c < cascades; c++ {
				wantExpr(fmt.Sprintf("probeOriginMod%d%s", c, ax), "mod", graph.E("sub", "param:probeOrigin"+ax, "param:water"+ax), sizes[c])
				if got := uniform("probe", fmt.Sprintf("om%s%d", ax, c)); got != fmt.Sprintf("param:probeOriginMod%d%s", c, ax) {
					t.Errorf("%s: probe om%s%d = %v", key, ax, c, got)
				}
			}
		}
	}
}

// The WebGL2 foam pass is 1,024 wide and reads the cascade and wake buffers at
// their own widths (each read buffer has a width uniform), so no graph copies a
// buffer to widen it.
func TestOceanGraphNoWidthCopies(t *testing.T) {
	gs, _ := loadGraphs(t)
	for _, key := range graphKeys {
		flat, err := graph.Expand(gs[key])
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range flat.Passes {
			if p.Type == "copy" {
				t.Errorf("%s: copy pass %s; a gather pass reads buffers at their own widths", key, p.Name)
			}
		}
	}
}
