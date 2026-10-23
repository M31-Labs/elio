package kernelsets

import (
	"fmt"
	"math"
	"math/bits"

	"m31labs.dev/elio/emit/ts"
	"m31labs.dev/elio/graph"
)

// OceanWithGraphs is Ocean plus the six ocean graphs, in the order
// webgpu/high, medium, low, then webgl2/high, medium, low. `elio emit ts --set
// ocean` emits this set.
func OceanWithGraphs() (ts.KernelSet, error) {
	set, err := Ocean()
	if err != nil {
		return ts.KernelSet{}, err
	}
	gs, err := oceanGraphs(set)
	if err != nil {
		return ts.KernelSet{}, err
	}
	for _, backend := range []string{graph.WebGPU, graph.WebGL2} {
		for _, tier := range oceanTiers {
			key := backend + "/" + tier.name
			set.Graphs = append(set.Graphs, ts.NamedGraph{Key: key, Graph: gs[key]})
		}
	}
	return set, nil
}

// The ocean's fixed conventions (spec "Wave model").
const (
	foamWidth    = 1024 // foam state texels per side
	foamCell     = 0.5  // metres per foam texel
	probeMax     = 64   // probe points
	probeStep    = 0.5  // metres, central-difference step for the probe normal
	tweenSeconds = 20   // presentation seconds a spectrum change takes (R10)
)

// The ship wake's fixed conventions. The window is wakeN x wakeN cells (the
// largest size the FFT kernels cover) of wakeCell metres, 4 by default: a 1,024 m
// window. The sponge band is wakeSponge cells wide with a peak damping rate of
// wakeSpongeRate per second. A hull is at full pressure from wakeFullSpeed m/s
// and at full foam from wakeFoamSpeed m/s. The window centre sits wakeBias of the
// window width behind the hull along its heading, so the trail gets 70% of the
// width.
//
// Step and stability. The leapfrog is stable for dt omega_max < 2 with
// omega_max^2 = g |k|_max and |k|_max cell = 2 pi sqrt(2) (wakeN/2 - 1) / wakeN
// (the Nyquist lines are zeroed). The graph caps the step at wakeDtCap seconds and
// the effective gravity at wakeGMaxPerCell x cell, which keeps dt omega_max at or
// below wakeStability (1.7) for any cell size. One step per frame is enough:
// at 60 Hz that allows a time scale of 12; above it the wake runs slow
// (a repeat of the pass sequence would restore it at the cost of a second
// set of 36 passes on webgl2).
const (
	wakeN           = 256
	wakeCellDefault = 4.0
	wakeSponge      = 24.0
	wakeSpongeRate  = 0.8
	wakeFullSpeed   = 3.0
	wakeFoamSpeed   = 5.0
	wakeBias        = 0.2
	wakeDtCap       = 0.2
	wakeStability   = 1.7
	wakeKMaxCell    = 2 * math.Pi * math.Sqrt2 * (wakeN/2 - 1) / wakeN
	wakeGMaxPerCell = wakeStability * wakeStability / (wakeDtCap * wakeDtCap * wakeKMaxCell)
)

// oceanCascade is one cascade: tile size in metres and band [kMin, kMax) in rad/m.
type oceanCascade struct{ size, kMin, kMax float64 }

var oceanCascades = [4]oceanCascade{
	{1600, 2 * math.Pi / 1600, 0.12},
	{250, 0.12, 0.80},
	{37, 0.80, 5.4},
	{5.3, 5.4, 37},
}

// tierResolution is the grid size of cascades 1 to 3, and the cascade count.
var oceanTiers = []struct {
	name     string
	n        int
	cascades int
}{{"high", 256, 4}, {"medium", 128, 4}, {"low", 64, 3}}

// spectrumParams are the params whose change reruns the spectrum passes.
var spectrumParams = []string{"windSpeed", "fetch", "gamma", "spread", "swell", "windDir", "swellDir", "seed", "loopPeriod"}

// OceanGraphs builds the ocean's pass graph for each backend and quality
// tier, keyed "webgpu/high" ... "webgl2/low". The graphs are validated against
// the Ocean kernel set.
//
// Params the app sets (value): windSpeed, fetch (km), gamma, spread, swell,
// windDirDeg and swellDirDeg (degrees as Scene3D authors them), seed,
// loopPeriod, lambda (choppiness), timeScale, driftX and driftZ (water drift,
// m/s), foamTau, foamThreshold, foamGain, probeCount, probePoints, probeOriginX
// and probeOriginZ (WebGPU; points are relative to probeOrigin, in world metres).
// The graph derives the rest: windDir and swellDir (travel angles
// pi/2 - deg pi/180), simTime (timeScale integrated), phaseTime (simTime mod
// loopPeriod), waterX and waterZ (drift integrated), and the foam window's
// origin and shift in whole texels in the water frame.
func OceanGraphs() (map[string]graph.Graph, error) {
	set, err := Ocean()
	if err != nil {
		return nil, err
	}
	return oceanGraphs(set)
}

func oceanGraphs(set ts.KernelSet) (map[string]graph.Graph, error) {
	return OceanGraphsForHull(set, "")
}

// OceanGraphsForHull is OceanGraphs with the hull pose (hullX, hullZ,
// hullHeading, hullSpeed) read by the runner from the scene node hullNode
// instead of set from the app; an empty hullNode keeps them plain value params.
func OceanGraphsForHull(set ts.KernelSet, hullNode string) (map[string]graph.Graph, error) {
	ks := set.GraphKernels()
	out := map[string]graph.Graph{}
	for _, backend := range []string{graph.WebGPU, graph.WebGL2} {
		for _, tier := range oceanTiers {
			g, err := buildOcean(backend, tier.name, tier.n, tier.cascades, ks, hullNode)
			if err != nil {
				return nil, err
			}
			out[backend+"/"+tier.name] = g
		}
	}
	return out, nil
}

func buildOcean(backend, tier string, n, cascades int, ks []graph.Kernel, hullNode string) (graph.Graph, error) {
	b := graph.NewBuilder("ocean", backend, tier, ks)
	gpu := backend == graph.WebGPU
	ref := graph.Ref
	deg := math.Pi / 180

	// Params, in evaluation order: values, frame inputs, then derived.
	for _, p := range []struct {
		name string
		def  float64
	}{{"windSpeed", 9}, {"fetch", 300}, {"gamma", 3.3}, {"spread", 8}, {"swell", 0}, {"windDirDeg", 0}, {"swellDirDeg", 0}} {
		b.TweenedValue(p.name, graph.F32, p.def, tweenSeconds)
	}
	b.Value("seed", graph.U32, 0).Value("loopPeriod", graph.F32, 1000)
	b.TweenedValue("lambda", graph.F32, 1, tweenSeconds)
	b.Value("timeScale", graph.F32, 1).Value("driftX", graph.F32, 0).Value("driftZ", graph.F32, 0)
	b.Value("foamTau", graph.F32, 4).Value("foamThreshold", graph.F32, 0.9).Value("foamGain", graph.F32, 2)
	b.Value("probeCount", graph.U32, 0)
	if gpu {
		// Probe points are relative to probeOrigin (world metres, set near the vessel).
		b.Value("probeOriginX", graph.F32, 0).Value("probeOriginZ", graph.F32, 0)
		b.BufferParam("probePoints", "vec4<f32>", probeMax)
	}
	b.Frame("dt", "dt").Frame("cameraX", "cameraX").Frame("cameraZ", "cameraZ")
	e := graph.E
	b.Expr("windDir", graph.F32, "sub", math.Pi/2, e("mul", ref("windDirDeg"), deg))
	b.Expr("swellDir", graph.F32, "sub", math.Pi/2, e("mul", ref("swellDirDeg"), deg))
	b.Accumulate("simTime", "timeScale")
	b.Expr("phaseTime", graph.F32, "mod", ref("simTime"), ref("loopPeriod"))
	b.Accumulate("waterX", "driftX").Accumulate("waterZ", "driftZ")
	// The surface samples cascade c at (xz - water) / size. The water offset
	// grows without bound, so the runner wraps it to each cascade's tile in
	// f64 and the surface binds the wrapped values. Every tier declares all four
	// cascades' params, so the tiers share one param list.
	for c := 0; c < len(oceanCascades); c++ {
		for _, ax := range []string{"X", "Z"} {
			n := fmt.Sprintf("waterMod%s%d", ax, c)
			b.Expr(n, graph.F32, "mod", ref("water"+ax), oceanCascades[c].size)
			b.ExportParams(n)
		}
	}
	for _, ax := range []string{"X", "Z"} {
		// The foam window origin in whole cells, in the water frame.
		b.Expr("foamOrigin"+ax, graph.I32, "sub", e("floor", e("div", e("sub", ref("camera"+ax), ref("water"+ax)), foamCell)), foamWidth/2)
		b.Delta("foamShift"+ax, graph.I32, "foamOrigin"+ax)
		// The foam kernel samples cascade c at the window origin wrapped into
		// the tile plus a local offset, so it forms no large coordinate in f32.
		for c := 0; c < len(oceanCascades); c++ {
			b.Expr(fmt.Sprintf("foamOriginMod%d%s", c, ax), graph.F32, "mod", e("mul", ref("foamOrigin"+ax), foamCell), oceanCascades[c].size)
		}
		if gpu {
			// The probe's point origin in the water frame, wrapped per cascade.
			for c := 0; c < len(oceanCascades); c++ {
				b.Expr(fmt.Sprintf("probeOriginMod%d%s", c, ax), graph.F32, "mod", e("sub", ref("probeOrigin"+ax), ref("water"+ax)), oceanCascades[c].size)
			}
		}
	}

	// Ship wake: the hull's pose and the wake window's derived origin, in the
	// water frame (see the wake section of OceanGraphs).
	if hullNode == "" {
		b.Value("hullX", graph.F32, 0).Value("hullZ", graph.F32, 0).Value("hullHeading", graph.F32, 0).Value("hullSpeed", graph.F32, 0)
	} else {
		b.Node("hullX", hullNode, "x").Node("hullZ", hullNode, "z").Node("hullHeading", hullNode, "heading").Node("hullSpeed", hullNode, "speed")
	}
	b.BoolValue("wakeEnabled", false)
	b.Value("hullLength", graph.F32, 200).Value("hullBeam", graph.F32, 30)
	b.Value("wakeCell", graph.F32, wakeCellDefault).Value("wakeDepth", graph.F32, 0.4).Value("wakeGravity", graph.F32, 9.81)
	b.Value("wakeAlpha", graph.F32, 0.02).Value("wakeMinLambdaCells", graph.F32, 8)
	b.Value("wakeFoamGain", graph.F32, 1).Value("wakeFoamTau", graph.F32, 45)
	b.Expr("wakeOn", graph.F32, "mul", ref("wakeEnabled"), 1.0)
	b.Expr("wakeFoamGainEff", graph.F32, "mul", ref("wakeEnabled"), ref("wakeFoamGain"))
	b.Delta("wakeEdge", graph.I32, "wakeEnabled")
	b.Expr("wakeReset", graph.U32, "max", ref("wakeEdge"), 0.0)
	b.Expr("wakeLength", graph.F32, "mul", ref("wakeCell"), wakeN)
	// The wake window's corner is the hull, less wakeBias of the window along
	// the heading, less half the window; origin and the hull's place in it are
	// whole cells and small offsets, all derived in float64.
	cell := ref("wakeCell")
	for _, ax := range []struct{ name, uv, dir string }{{"X", "U", "sin"}, {"Z", "V", "cos"}} {
		a := ax.name
		b.Expr("hullWater"+a, graph.F32, "sub", ref("hull"+a), ref("water"+a))
		corner := e("sub", e("sub", ref("hullWater"+a), e("mul", e(ax.dir, ref("hullHeading")), e("mul", cell, wakeBias*wakeN))), e("mul", cell, wakeN/2))
		b.Expr("wakeOrigin"+a, graph.I32, "floor", e("div", corner, cell))
		b.Delta("wakeShift"+a, graph.I32, "wakeOrigin"+a)
		originM := e("mul", ref("wakeOrigin"+a), cell)
		b.Expr("wakeHull"+ax.uv, graph.F32, "sub", ref("hullWater"+a), originM)
		b.Expr("wakeOriginMod"+a, graph.I32, "mod", ref("wakeOrigin"+a), wakeN)
		b.Expr("wakeRel"+a, graph.F32, "sub", e("mul", ref("foamOrigin"+a), foamCell), originM)
		b.Expr("wakeCamRel"+a, graph.F32, "sub", e("add", ref("water"+a), originM), ref("camera"+a))
	}
	// Gravity and step. The transverse wavelength is 2 pi V^2 / g_eff, which is
	// unresolved on the grid at low speed. So g_eff = min(g, 2 pi V^2 / (wakeMinLambdaCells
	// cell)), at least 0.05: the wavelength is the real one when it is at least
	// wakeMinLambdaCells cells (V above 7.1 m/s, 14 knots, at 4 m cells) and is
	// held at wakeMinLambdaCells cells below that, so a slow hull still shows a
	// Kelvin wedge, with longer waves than the real ones. g_eff is also capped by
	// the stability limit. The step is the frame time (scaled) up to wakeDtCap.
	speedSq := e("mul", ref("hullSpeed"), ref("hullSpeed"))
	b.Expr("wakeGEff", graph.F32, "max",
		e("min", e("min", ref("wakeGravity"), e("mul", speedSq, e("div", 2*math.Pi, e("mul", ref("wakeMinLambdaCells"), cell)))), e("mul", cell, wakeGMaxPerCell)),
		0.05)
	b.Expr("wakeDt", graph.F32, "min", e("mul", ref("dt"), ref("timeScale")), wakeDtCap)
	// The pressure patch is half a transverse wavelength long (at least two
	// cells) and no longer than the hull; its beam is 0.67 of its length, at
	// least 1.5 cells and no wider than the hull.
	b.Expr("wakeSrcLength", graph.F32, "max", e("min", ref("hullLength"), e("div", e("mul", speedSq, math.Pi), ref("wakeGEff"))), e("mul", cell, 2.0))
	b.Expr("wakeSrcBeam", graph.F32, "max", e("min", ref("hullBeam"), e("mul", ref("wakeSrcLength"), 0.67)), e("mul", cell, 1.5))
	// The surface samples the wake texture at ((xz - camera - wakeCamRel) / wakeCell
	// + wakeOriginMod) / 256 with wrap addressing, inside the window only.
	b.ExportParams("wakeCamRelX", "wakeCamRelZ", "wakeOriginModX", "wakeOriginModZ", "wakeCell", "wakeOn")

	size := func(c int) int {
		if c == 0 {
			return 64
		}
		return n
	}
	name := func(prefix string, c int) string { return fmt.Sprintf("%s%d", prefix, c) }

	// Where a pass runs: a workgroup count (WebGPU) or a render target (WebGL2).
	place := func(p graph.Pass, kernel string, w, h int) graph.Pass {
		if gpu {
			p.Dispatch = b.Workgroups(kernel, p.Entry, w*h)
		} else {
			p.Target = &graph.Target{Width: w, Height: h, ElioWidth: w}
		}
		return p
	}
	// Mips: WebGPU downsamples level by level; WebGL2 calls generateMipmap.
	mipChain := func(pre, tex string, w, mips int, when string) {
		if !gpu {
			b.Pass(graph.Pass{Name: pre + tex + ".mips", Type: "mipmap", Resource: tex, When: when})
			return
		}
		// One repeat over the levels: level l+1 is written from level l. The
		// dispatch is sized for level 1; smaller levels leave the extra
		// workgroups idle (the kernel returns past its destination size).
		half := max(w>>1, 1)
		b.Pass(graph.Pass{Name: pre + tex + ".mips", Type: "repeat", Count: mips - 1, Index: "l", When: when, Body: []graph.Pass{
			{Name: "mip", Kernel: "ocean_mip",
				Dispatch: b.Workgroups("ocean_mip", "ocean_mip", half*half),
				Bindings: []graph.Binding{
					{Name: "src", Resource: tex, LevelAt: &graph.IndexExpr{Index: "l"}},
					{Name: "dst", Resource: tex, LevelAt: &graph.IndexExpr{Index: "l", Offset: 1}}},
				Uniforms: []graph.Uniform{b.Uniform("ocean_mip", "mip", map[string]any{"srcLevel": 0,
					"dstWidth": graph.IndexExpr{Index: "l", Offset: 1, Shr: w}, "dstHeight": graph.IndexExpr{Index: "l", Offset: 1, Shr: w}})}}}})
	}
	buf := func(rname string, elem string, count, width int) { b.Buffer(rname, elem, count, width) }

	for c := 0; c < cascades; c++ {
		nc, cc := size(c), oceanCascades[c]
		area := nc * nc
		buf(name("h0_", c), "vec4<f32>", area, nc)
		buf(name("omega_", c), "f32", area, nc)
		buf(name("fftA_", c), "vec4<f32>", area, nc)
		buf(name("fftB_", c), "vec4<f32>", area, nc)
		if !gpu {
			buf(name("fftA2_", c), "vec4<f32>", area, nc)
			buf(name("fftB2_", c), "vec4<f32>", area, nc)
		}
		mips := graph.MipCount(nc, nc)
		b.Texture(name("dispTex", c), "rgba16float", nc, nc, mips)
		b.Texture(name("slopeTex", c), "rgba16float", nc, nc, mips)
		pre := fmt.Sprintf("c%d.", c)

		// Spectrum, when a spectrum input changed.
		sp := graph.Pass{Name: pre + "spectrum", Kernel: "ocean_spectrum",
			When:     graph.Changed(spectrumParams...),
			Bindings: []graph.Binding{{Name: "h0", Resource: name("h0_", c)}, {Name: "omega", Resource: name("omega_", c)}},
			Uniforms: []graph.Uniform{b.Uniform("ocean_spectrum", "spec", map[string]any{
				"n": nc, "cascade": c, "seed": ref("seed"), "windSpeed": ref("windSpeed"), "fetchKm": ref("fetch"),
				"gamma": ref("gamma"), "spread": ref("spread"), "swell": ref("swell"), "windDir": ref("windDir"),
				"swellDir": ref("swellDir"), "tile": cc.size, "kMin": cc.kMin, "kMax": cc.kMax, "loopPeriod": ref("loopPeriod")})}}
		b.Pass(place(sp, "ocean_spectrum", nc, nc))

		// Evolve to the phase time.
		ev := graph.Pass{Name: pre + "evolve", Kernel: "ocean_evolve",
			Bindings: []graph.Binding{{Name: "h0", Resource: name("h0_", c)}, {Name: "omega", Resource: name("omega_", c)},
				{Name: "outA", Resource: name("fftA_", c)}, {Name: "outB", Resource: name("fftB_", c)}},
			Uniforms: []graph.Uniform{b.Uniform("ocean_evolve", "ev", map[string]any{
				"n": nc, "t": ref("phaseTime"), "tile": cc.size, "loopPeriod": ref("loopPeriod")})}}
		b.Pass(place(ev, "ocean_evolve", nc, nc))

		// Inverse FFT of A and B.
		fk := fmt.Sprintf("fft_compute_%d", nc)
		gk := fmt.Sprintf("fft_gather_%d", nc)
		if gpu {
			for _, x := range []string{"A", "B"} {
				for _, entry := range []string{"fft_rows", "fft_cols"} {
					b.Pass(graph.Pass{Name: pre + "fft" + x + "." + entry, Kernel: fk, Entry: notPrimary(entry),
						Dispatch: &[3]int{nc, 1, 1}, Bindings: []graph.Binding{{Name: "data", Resource: name("fft"+x+"_", c)}}})
				}
			}
		} else {
			// One stage per pass, rows then columns, A and B together; the
			// buffers alternate with their scratch twins, which the repeat's
			// swap expresses.
			for axis, label := range []string{"rows", "cols"} {
				var body []graph.Pass
				for _, x := range []string{"A", "B"} {
					body = append(body, place(graph.Pass{Name: x, Kernel: gk,
						Bindings: []graph.Binding{{Name: "src", Resource: name("fft"+x+"_", c)}, {Name: "dst", Resource: name("fft"+x+"2_", c)}},
						Uniforms: []graph.Uniform{b.Uniform(gk, "fft", map[string]any{"n": nc, "stage": ref("s"), "axis": axis})}}, gk, nc, nc))
				}
				b.Pass(graph.Pass{Name: pre + "fft." + label, Type: "repeat", Count: bitsLen(nc), Index: "s",
					Swap: [][2]string{{name("fftA_", c), name("fftA2_", c)}, {name("fftB_", c), name("fftB2_", c)}}, Body: body})
			}
		}

		// Assemble into the cascade textures, then mips.
		as := graph.Pass{Name: pre + "assemble", Kernel: "ocean_assemble",
			Bindings: []graph.Binding{{Name: "fieldA", Resource: name("fftA_", c)}, {Name: "fieldB", Resource: name("fftB_", c)},
				{Name: "disp", Resource: name("dispTex", c)}, {Name: "slope", Resource: name("slopeTex", c)}},
			Uniforms: []graph.Uniform{b.Uniform("ocean_assemble", "assem", map[string]any{"n": nc, "lambda": ref("lambda")})}}
		b.Pass(place(as, "ocean_assemble", nc, nc))
		for _, tex := range []string{name("dispTex", c), name("slopeTex", c)} {
			mipChain(pre, tex, nc, mips, "")
		}
		b.Export(name("disp", c), name("dispTex", c)).Export(name("slope", c), name("slopeTex", c))
	}

	// Ship wake (when wakeEnabled): a spectral free surface in a 256 x 256
	// toroidal window of the water frame. Per frame: wake_source (clear wrapped-in
	// cells, hull pressure, foam turbulence) -> inverse FFT -> wake_symbol (|k|)
	// -> inverse FFT -> wake_step (leapfrog, sponge, texture). The FFT buffers
	// wakeF and wakeG are each other's scratch on webgl2, so two buffers serve
	// both transforms; the operator output ends in wakeG on both backends.
	wakeWhen := "param:wakeEnabled"
	wakeCount := wakeN * wakeN
	for _, n := range []string{"wakeA", "wakeB", "wakeF", "wakeG", "wakeC"} {
		buf(n, "vec4<f32>", wakeCount, wakeN)
	}
	wakeMips := graph.MipCount(wakeN, wakeN)
	b.Texture("wakeTex", "rgba16float", wakeN, wakeN, wakeMips)
	wp := func(p graph.Pass, kernel string) graph.Pass {
		p.When = wakeWhen
		return place(p, kernel, wakeN, wakeN)
	}
	b.Pass(wp(graph.Pass{Name: "wake.source", Kernel: "wake_source",
		Bindings: []graph.Binding{{Name: "wakeIn", Resource: "wakeA"}, {Name: "fftOut", Resource: "wakeF"}, {Name: "cleanOut", Resource: "wakeC"}},
		Uniforms: []graph.Uniform{b.Uniform("wake_source", "wake", map[string]any{
			"width": wakeN, "reset": ref("wakeReset"),
			"originX": ref("wakeOriginX"), "originZ": ref("wakeOriginZ"), "shiftX": ref("wakeShiftX"), "shiftZ": ref("wakeShiftZ"),
			"cell": ref("wakeCell"), "dt": ref("wakeDt"), "tau": ref("wakeFoamTau"), "depth": ref("wakeDepth"),
			"hullU": ref("wakeHullU"), "hullV": ref("wakeHullV"), "heading": ref("hullHeading"), "speed": ref("hullSpeed"),
			"hullLen": ref("hullLength"), "hullBeam": ref("hullBeam"), "srcLength": ref("wakeSrcLength"), "srcBeam": ref("wakeSrcBeam"),
			"fullSpeed": wakeFullSpeed, "foamSpeed": wakeFoamSpeed})}}, "wake_source"))
	wfk, wgk := fmt.Sprintf("fft_compute_%d", wakeN), fmt.Sprintf("fft_gather_%d", wakeN)
	wakeFFT := func(label, src, dst string) {
		if gpu {
			for _, entry := range []string{"fft_rows", "fft_cols"} {
				b.Pass(graph.Pass{Name: "wake." + label + "." + entry, Kernel: wfk, Entry: notPrimary(entry), When: wakeWhen,
					Dispatch: &[3]int{wakeN, 1, 1}, Bindings: []graph.Binding{{Name: "data", Resource: src}}})
			}
			return
		}
		for axis, ax := range []string{"rows", "cols"} {
			body := []graph.Pass{place(graph.Pass{Name: "stage", Kernel: wgk,
				Bindings: []graph.Binding{{Name: "src", Resource: src}, {Name: "dst", Resource: dst}},
				Uniforms: []graph.Uniform{b.Uniform(wgk, "fft", map[string]any{"n": wakeN, "stage": ref("s"), "axis": axis})}}, wgk, wakeN, wakeN)}
			b.Pass(graph.Pass{Name: "wake." + label + "." + ax, Type: "repeat", Count: bitsLen(wakeN), Index: "s", When: wakeWhen,
				Swap: [][2]string{{src, dst}}, Body: body})
		}
	}
	wakeFFT("fft1", "wakeF", "wakeG")
	b.Pass(wp(graph.Pass{Name: "wake.symbol", Kernel: "wake_symbol",
		Bindings: []graph.Binding{{Name: "fftIn", Resource: "wakeF"}, {Name: "fftOut", Resource: "wakeG"}},
		Uniforms: []graph.Uniform{b.Uniform("wake_symbol", "sym", map[string]any{"n": wakeN, "window": ref("wakeLength")})}}, "wake_symbol"))
	wakeFFT("fft2", "wakeG", "wakeF")
	b.Pass(wp(graph.Pass{Name: "wake.step", Kernel: "wake_step",
		Bindings: []graph.Binding{{Name: "qIn", Resource: "wakeG"}, {Name: "cleanIn", Resource: "wakeC"},
			{Name: "stateOut", Resource: "wakeB"}, {Name: "wakeTex", Resource: "wakeTex"}},
		Uniforms: []graph.Uniform{b.Uniform("wake_step", "wstep", map[string]any{
			"width": wakeN, "originX": ref("wakeOriginX"), "originZ": ref("wakeOriginZ"),
			"sponge": wakeSponge, "spongeRate": wakeSpongeRate,
			"dt": ref("wakeDt"), "gravity": ref("wakeGEff"), "alpha": ref("wakeAlpha"), "cell": ref("wakeCell")})}}, "wake_step"))
	mipChain("wake.", "wakeTex", wakeN, wakeMips, wakeWhen)
	b.Swap("wakeA", "wakeB").Export("wake", "wakeTex")

	// Foam: a toroidal state pair, swapped after each frame, and its texture.
	foamArea := foamWidth * foamWidth
	buf("foamA", "f32", foamArea, foamWidth)
	buf("foamB", "f32", foamArea, foamWidth)
	foamMips := graph.MipCount(foamWidth, foamWidth)
	b.Texture("foamTex", "rgba16float", foamWidth, foamWidth, foamMips)
	fu := map[string]any{
		"width": foamWidth, "cascades": cascades,
		"originX": ref("foamOriginX"), "originZ": ref("foamOriginZ"), "shiftX": ref("foamShiftX"), "shiftZ": ref("foamShiftZ"),
		"dt": ref("dt"), "tau": ref("foamTau"), "threshold": ref("foamThreshold"), "gain": ref("foamGain"),
		"lambda": ref("lambda"), "cell": foamCell,
	}
	for i := 0; i < 4; i++ {
		c := min(i, cascades-1)
		fu[fmt.Sprintf("omX%d", i)] = ref(fmt.Sprintf("foamOriginMod%dX", c))
		fu[fmt.Sprintf("omZ%d", i)] = ref(fmt.Sprintf("foamOriginMod%dZ", c))
	}
	// Foam follows the wake's turbulence (zero gain unless wakeEnabled).
	fu["wakeWidth"], fu["wakeGain"], fu["wakeCell"] = wakeN, ref("wakeFoamGainEff"), ref("wakeCell")
	fu["wakeRelX"], fu["wakeRelZ"] = ref("wakeRelX"), ref("wakeRelZ")
	fu["wakeOriginX"], fu["wakeOriginZ"] = ref("wakeOriginModX"), ref("wakeOriginModZ")
	// WebGL2: the foam pass is foamWidth wide and reads each buffer at that
	// width, so it reads copies made by copy passes (the same elements laid out
	// foamWidth texels wide). WebGPU reads the buffers directly.
	foamRead := func(src string, count int) string {
		if gpu {
			return src
		}
		view := src + "_w"
		buf(view, "vec4<f32>", count, foamWidth)
		b.Pass(graph.Pass{Name: "foam.copy." + src, Type: "copy", From: src, To: view, Width: foamWidth})
		return view
	}
	foamBind := []graph.Binding{{Name: "foamIn", Resource: "foamA"}}
	probeBind := []graph.Binding{}
	var views [4]string
	for i := 0; i < 4; i++ {
		c := min(i, cascades-1) // an unused cascade slot repeats the last real one
		nc := size(c)
		fu[fmt.Sprintf("n%d", i)], fu[fmt.Sprintf("size%d", i)] = nc, oceanCascades[c].size
		if i < cascades {
			views[i] = foamRead(name("fftB_", c), nc*nc)
		} else {
			views[i] = views[c]
		}
		foamBind = append(foamBind, graph.Binding{Name: fmt.Sprintf("fieldB%d", i), Resource: views[i]})
		probeBind = append(probeBind, graph.Binding{Name: fmt.Sprintf("disp%d", i), Resource: name("fftA_", c)})
	}
	foamBind = append(foamBind, graph.Binding{Name: "foamOut", Resource: "foamB"}, graph.Binding{Name: "foamTex", Resource: "foamTex"}, graph.Binding{Name: "wakeIn", Resource: foamRead("wakeB", wakeN*wakeN)})
	b.Pass(place(graph.Pass{Name: "foam", Kernel: "ocean_foam",
		Bindings: foamBind, Uniforms: []graph.Uniform{b.Uniform("ocean_foam", "foam", fu)}}, "ocean_foam", foamWidth, foamWidth))
	mipChain("", "foamTex", foamWidth, foamMips, "")
	b.Swap("foamA", "foamB").Export("foam", "foamTex")

	// Probe (WebGPU only): sample the surface at the app's points for readback.
	if gpu {
		b.FedBuffer("probePoints", "vec4<f32>", probeMax, 0, "probePoints")
		b.Buffer("probeOut", "vec4<f32>", probeMax, 0)
		pu := map[string]any{"count": ref("probeCount"), "cascades": cascades, "t": ref("simTime"), "normalStep": probeStep, "lambda": ref("lambda")}
		for i := 0; i < 4; i++ {
			c := min(i, cascades-1)
			pu[fmt.Sprintf("n%d", i)], pu[fmt.Sprintf("size%d", i)] = size(c), oceanCascades[c].size
			pu[fmt.Sprintf("omX%d", i)] = ref(fmt.Sprintf("probeOriginMod%dX", c))
			pu[fmt.Sprintf("omZ%d", i)] = ref(fmt.Sprintf("probeOriginMod%dZ", c))
		}
		bind := append([]graph.Binding{{Name: "points", Resource: "probePoints"}}, probeBind...)
		bind = append(bind, graph.Binding{Name: "out", Resource: "probeOut"})
		b.Pass(graph.Pass{Name: "probe", Kernel: "ocean_probe",
			Dispatch: b.Workgroups("ocean_probe", "ocean_probe", probeMax), Bindings: bind,
			Uniforms: []graph.Uniform{b.Uniform("ocean_probe", "probe", pu)}})
		b.Export("probe", "probeOut")
	}
	return b.Build()
}

// bitsLen is log2 of a power of two.
func bitsLen(n int) int { return bits.TrailingZeros(uint(n)) }

// notPrimary leaves out the FFT compute kernel's primary entry (fft_rows), which
// a pass gets by default; fft_cols is written out.
func notPrimary(entry string) string {
	if entry == "fft_rows" {
		return ""
	}
	return entry
}
