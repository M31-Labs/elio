package kernelsets

import (
	"math"
	"sync"
	"testing"

	"m31labs.dev/elio/graph"
	"m31labs.dev/elio/internal/wakesim"
)

// evalExprs evaluates a graph's value and expression params (frame, delta,
// accumulate and node params read 0) with the given overrides.
func evalExprs(g graph.Graph, set map[string]float64) map[string]float64 {
	v := map[string]float64{}
	var eval func(op string, a []any) float64
	arg := func(a any) float64 {
		switch x := a.(type) {
		case string:
			return v[x[len("param:"):]]
		case graph.Expr:
			return eval(x.Op, x.Args)
		case int:
			return float64(x)
		}
		return a.(float64)
	}
	eval = func(op string, a []any) float64 {
		switch op {
		case "add":
			return arg(a[0]) + arg(a[1])
		case "sub":
			return arg(a[0]) - arg(a[1])
		case "mul":
			return arg(a[0]) * arg(a[1])
		case "div":
			return arg(a[0]) / arg(a[1])
		case "mod":
			x, m := arg(a[0]), arg(a[1])
			return x - m*math.Floor(x/m)
		case "min":
			return math.Min(arg(a[0]), arg(a[1]))
		case "max":
			return math.Max(arg(a[0]), arg(a[1]))
		case "floor":
			return math.Floor(arg(a[0]))
		case "sin":
			return math.Sin(arg(a[0]))
		case "cos":
			return math.Cos(arg(a[0]))
		}
		panic("unknown op " + op)
	}
	for _, p := range g.Params {
		switch p.Source.Kind {
		case graph.SourceValue:
			v[p.Name] = p.Source.Value
		case graph.SourceExpr:
			v[p.Name] = eval(p.Source.Op, p.Source.Args)
		}
		if x, ok := set[p.Name]; ok {
			v[p.Name] = x
		}
	}
	return v
}

const knot = 0.514444

// The graph's gravity and pressure-patch rules, for a 125 m and a 300 m hull at
// 8, 15 and 22 knots (4 m cells): the transverse wavelength stays at least 8
// cells, the pressure patch is half a wavelength (so the hull length does not
// change the wedge), and the exact-operator wake in the matching dimensionless
// units has a half-angle within 3 degrees of 19.47. (The kernels themselves are
// run at two speeds in stdlib.TestKelvinWedgeOnKernels.)
func TestOceanWakeGraphRulesGiveKelvinAngle(t *testing.T) {
	gs, _ := loadGraphs(t)
	g := gs["webgpu/high"]
	type result struct {
		hull, kn         float64
		lambda, src, ang float64
	}
	hulls, speeds := []float64{125, 300}, []float64{8, 15, 22}
	out := make([]result, len(hulls)*len(speeds))
	var wg sync.WaitGroup
	for hi, hull := range hulls {
		for si, kn := range speeds {
			wg.Add(1)
			go func() {
				defer wg.Done()
				v := evalExprs(g, map[string]float64{"hullSpeed": kn * knot, "hullLength": hull, "hullBeam": hull / 6.5})
				cell := v["wakeCell"]
				lambda := 2 * math.Pi * v["hullSpeed"] * v["hullSpeed"] / v["wakeGEff"] // metres
				vSim := math.Sqrt(lambda / cell / (2 * math.Pi))                        // sqrt(g cell) units
				ang, n := wakesim.ReferenceAngle(vSim, v["wakeSrcLength"]/cell, v["wakeSrcBeam"]/cell)
				out[hi*len(speeds)+si] = result{hull, kn, lambda / cell, v["wakeSrcLength"] / cell, ang}
				if n < 8 {
					t.Errorf("%g m at %g kn: only %d cross-sections", hull, kn, n)
				}
				if lambda/cell < 8-1e-9 {
					t.Errorf("%g m at %g kn: transverse wavelength %.1f cells, want >= 8", hull, kn, lambda/cell)
				}
			}()
		}
	}
	wg.Wait()
	for _, r := range out {
		t.Logf("%3.0f m hull at %2.0f kn: transverse wavelength %4.1f cells (%.0f m), pressure patch %4.1f cells, half-angle %.1f deg", r.hull, r.kn, r.lambda, r.lambda*4, r.src, r.ang)
		if math.Abs(r.ang-19.47) > 3 {
			t.Errorf("%g m at %g kn: half-angle %.2f deg, want 19.47 +- 3", r.hull, r.kn, r.ang)
		}
	}
	// The patch, hence the wedge, does not depend on the hull length once the hull
	// is longer than half a wavelength.
	if out[0].src != out[3].src && out[0].lambda/2 < 125/4 {
		t.Errorf("patch length depends on the hull: %g vs %g cells", out[0].src, out[3].src)
	}
}
