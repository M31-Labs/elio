package kernelsets

import (
	"strings"
	"testing"

	"m31labs.dev/elio/graph"
)

func passByName(t *testing.T, g graph.Graph, name string) graph.Pass {
	t.Helper()
	flat, err := graph.Expand(g)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range flat.Passes {
		if p.Name == name {
			return p
		}
	}
	t.Fatalf("%s/%s: no pass %q", g.Backend, g.Tier, name)
	return graph.Pass{}
}

func paramsOf(g graph.Graph) map[string]graph.Param {
	m := map[string]graph.Param{}
	for _, p := range g.Params {
		m[p.Name] = p
	}
	return m
}

func TestOceanGraphWake(t *testing.T) {
	gs, _ := loadGraphs(t)
	for _, key := range graphKeys {
		g := gs[key]
		flat, err := graph.Expand(g)
		if err != nil {
			t.Fatal(err)
		}
		ps := paramsOf(g)
		if p := ps["wakeEnabled"]; p.Type != graph.Bool || p.Source.Kind != graph.SourceValue {
			t.Errorf("%s: wakeEnabled = %+v", key, p)
		}
		var wake []graph.Pass
		for _, p := range flat.Passes {
			if strings.HasPrefix(p.Name, "wake.") {
				wake = append(wake, p)
				if p.When != "param:wakeEnabled" {
					t.Errorf("%s: %s runs %q, want param:wakeEnabled", key, p.Name, p.When)
				}
			}
		}
		if len(wake) == 0 {
			t.Fatalf("%s: no wake passes", key)
		}
		// Resources: 256^2 state pair, FFT pair, clean state, 256^2 texture with mips.
		res := map[string]graph.Resource{}
		for _, r := range g.Resources {
			res[r.Name] = r
		}
		for _, n := range []string{"wakeA", "wakeB", "wakeF", "wakeG", "wakeC"} {
			if r := res[n]; r.ElementType != "vec4<f32>" || r.Count != 256*256 || (g.Backend == "webgl2" && r.Width != 256) {
				t.Errorf("%s: resource %s = %+v", key, n, r)
			}
		}
		if r := res["wakeTex"]; r.Format != "rgba16float" || r.Width != 256 || r.Height != 256 || r.MipLevels != 9 {
			t.Errorf("%s: wakeTex = %+v", key, r)
		}
		if g.Exports["wake"] != "wakeTex" {
			t.Errorf("%s: export wake = %q", key, g.Exports["wake"])
		}
		swapped := false
		for _, s := range g.FrameSwaps {
			swapped = swapped || (s == [2]string{"wakeA", "wakeB"})
		}
		if !swapped {
			t.Errorf("%s: no wakeA/wakeB frame swap", key)
		}
		// The step reads the operator output (wakeG) and writes the new state and texture.
		step := passByName(t, g, "wake.step")
		bound := map[string]string{}
		for _, b := range step.Bindings {
			bound[b.Name] = b.Resource
		}
		if bound["qIn"] != "wakeG" || bound["cleanIn"] != "wakeC" || bound["stateOut"] != "wakeB" || bound["wakeTex"] != "wakeTex" {
			t.Errorf("%s: wake.step bindings %v", key, bound)
		}
		if src := passByName(t, g, "wake.source"); len(src.Bindings) != 3 {
			t.Errorf("%s: wake.source bindings %v", key, src.Bindings)
		}
		// Foam reads the fresh wake state and follows it only while the wake is enabled.
		foam := passByName(t, g, "foam")
		var foamWake string
		for _, b := range foam.Bindings {
			if b.Name == "wakeIn" {
				foamWake = b.Resource
			}
		}
		if foamWake != "wakeB" {
			t.Errorf("%s: foam wakeIn bound to %q, want wakeB", key, foamWake)
		}
		if got := foam.Uniforms[0].Values["wakeGain"]; got != "param:wakeFoamGainEff" {
			t.Errorf("%s: foam wakeGain = %v", key, got)
		}
		if p := ps["wakeFoamGainEff"]; p.Source.Op != "mul" || p.Source.Args[0] != "param:wakeEnabled" {
			t.Errorf("%s: wakeFoamGainEff = %+v", key, p.Source)
		}
		for _, n := range []string{"wakeCamRelX", "wakeCamRelZ", "wakeOriginModX", "wakeOriginModZ", "wakeCell", "wakeOn"} {
			found := false
			for _, e := range g.ExportParams {
				found = found || e == n
			}
			if !found {
				t.Errorf("%s: %s is not an export param", key, n)
			}
		}
		// Gravity floor and step cap, as documented in the graph source.
		if p := ps["wakeDt"]; p.Source.Op != "min" || p.Source.Args[1] != 0.2 {
			t.Errorf("%s: wakeDt = %+v, want min(dt scaled, 0.2)", key, p.Source)
		}
		if p := ps["wakeGEff"]; p.Source.Op != "max" || p.Source.Args[1] != 0.05 {
			t.Errorf("%s: wakeGEff = %+v", key, p.Source)
		}
		t.Logf("%s: %d wake passes, graph memory %.1f MB", key, len(wake), float64(graph.MemoryBytes(g))/1e6)
	}
}

// The hull pose can come from a scene node instead of plain value params.
func TestOceanGraphsHullFromNode(t *testing.T) {
	set, err := Ocean()
	if err != nil {
		t.Fatal(err)
	}
	gs, err := OceanGraphsForHull(set, "ship")
	if err != nil {
		t.Fatal(err)
	}
	for key, g := range gs {
		ps := paramsOf(g)
		for field, name := range map[string]string{"x": "hullX", "z": "hullZ", "heading": "hullHeading", "speed": "hullSpeed"} {
			s := ps[name].Source
			if s.Kind != graph.SourceNode || s.Node != "ship" || s.Field != field {
				t.Errorf("%s: %s source %+v", key, name, s)
			}
		}
	}
}

// With the wake enabled the foam, wake and surface params are all derived in
// float64 from small numbers: the window corner and the hull's place in it.
func TestOceanGraphWakeWindowParams(t *testing.T) {
	gs, _ := loadGraphs(t)
	ps := paramsOf(gs["webgpu/high"])
	for _, c := range []struct{ name, op string }{
		{"wakeHullU", "sub"}, {"wakeHullV", "sub"}, {"wakeOriginX", "floor"}, {"wakeOriginZ", "floor"},
		{"wakeOriginModX", "mod"}, {"wakeRelX", "sub"}, {"wakeCamRelX", "sub"}, {"wakeShiftX", ""}, {"wakeReset", "max"},
	} {
		p, ok := ps[c.name]
		if !ok {
			t.Errorf("no param %s", c.name)
			continue
		}
		if c.op != "" && p.Source.Op != c.op {
			t.Errorf("%s op %s, want %s", c.name, p.Source.Op, c.op)
		}
	}
	if ps["wakeShiftX"].Source.Kind != graph.SourceDelta || ps["wakeShiftX"].Source.Of != "param:wakeOriginX" {
		t.Errorf("wakeShiftX = %+v", ps["wakeShiftX"].Source)
	}
	if ps["wakeReset"].Type != graph.U32 {
		t.Errorf("wakeReset type %s", ps["wakeReset"].Type)
	}
}
