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
// m/s), foamTau, foamThreshold, foamGain, probeCount and probePoints (WebGPU).
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
	ks := set.GraphKernels()
	out := map[string]graph.Graph{}
	for _, backend := range []string{graph.WebGPU, graph.WebGL2} {
		for _, tier := range oceanTiers {
			g, err := buildOcean(backend, tier.name, tier.n, tier.cascades, ks)
			if err != nil {
				return nil, err
			}
			out[backend+"/"+tier.name] = g
		}
	}
	return out, nil
}

func buildOcean(backend, tier string, n, cascades int, ks []graph.Kernel) (graph.Graph, error) {
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
		b.BufferParam("probePoints", "vec4<f32>", probeMax)
	}
	b.Frame("dt", "dt").Frame("cameraX", "cameraX").Frame("cameraZ", "cameraZ")
	b.Expr("windDirRad", graph.F32, "mul", ref("windDirDeg"), deg)
	b.Expr("windDir", graph.F32, "sub", math.Pi/2, ref("windDirRad"))
	b.Expr("swellDirRad", graph.F32, "mul", ref("swellDirDeg"), deg)
	b.Expr("swellDir", graph.F32, "sub", math.Pi/2, ref("swellDirRad"))
	b.Accumulate("simTime", "timeScale")
	b.Expr("phaseTime", graph.F32, "mod", ref("simTime"), ref("loopPeriod"))
	b.Accumulate("waterX", "driftX").Accumulate("waterZ", "driftZ")
	// The surface samples cascade c at (xz - water) / size. The water offset
	// grows without bound, so the runner wraps it to each cascade's tile in
	// f64 and the surface binds the wrapped values.
	for c := 0; c < cascades; c++ {
		for _, ax := range []string{"X", "Z"} {
			n := fmt.Sprintf("waterMod%s%d", ax, c)
			b.Expr(n, graph.F32, "mod", ref("water"+ax), oceanCascades[c].size)
			b.ExportParams(n)
		}
	}
	for _, ax := range []string{"X", "Z"} {
		b.Expr("foamRel"+ax, graph.F32, "sub", ref("camera"+ax), ref("water"+ax))
		b.Expr("foamTexel"+ax, graph.F32, "div", ref("foamRel"+ax), foamCell)
		b.Expr("foamCell"+ax, graph.F32, "floor", ref("foamTexel"+ax))
		b.Expr("foamOrigin"+ax, graph.I32, "sub", ref("foamCell"+ax), foamWidth/2)
		b.Delta("foamShift"+ax, graph.I32, "foamOrigin"+ax)
	}

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
	mipChain := func(pre, tex string, w, mips int) {
		if !gpu {
			b.Pass(graph.Pass{Name: pre + tex + ".mips", Type: "mipmap", Resource: tex, When: "always"})
			return
		}
		// One repeat over the levels: level l+1 is written from level l. The
		// dispatch is sized for level 1; smaller levels leave the extra
		// workgroups idle (the kernel returns past its destination size).
		half := max(w>>1, 1)
		b.Pass(graph.Pass{Name: pre + tex + ".mips", Type: "repeat", Count: mips - 1, Index: "l", When: "always", Body: []graph.Pass{
			{Name: "mip", Type: "kernel", Kernel: "ocean_mip", Entry: "ocean_mip", When: "always",
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
		sp := graph.Pass{Name: pre + "spectrum", Type: "kernel", Kernel: "ocean_spectrum", Entry: "ocean_spectrum",
			When:     graph.Changed(spectrumParams...),
			Bindings: []graph.Binding{{Name: "h0", Resource: name("h0_", c)}, {Name: "omega", Resource: name("omega_", c)}},
			Uniforms: []graph.Uniform{b.Uniform("ocean_spectrum", "spec", map[string]any{
				"n": nc, "cascade": c, "seed": ref("seed"), "windSpeed": ref("windSpeed"), "fetchKm": ref("fetch"),
				"gamma": ref("gamma"), "spread": ref("spread"), "swell": ref("swell"), "windDir": ref("windDir"),
				"swellDir": ref("swellDir"), "tile": cc.size, "kMin": cc.kMin, "kMax": cc.kMax, "loopPeriod": ref("loopPeriod")})}}
		b.Pass(place(sp, "ocean_spectrum", nc, nc))

		// Evolve to the phase time.
		ev := graph.Pass{Name: pre + "evolve", Type: "kernel", Kernel: "ocean_evolve", Entry: "ocean_evolve", When: "always",
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
					b.Pass(graph.Pass{Name: pre + "fft" + x + "." + entry, Type: "kernel", Kernel: fk, Entry: entry, When: "always",
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
					body = append(body, place(graph.Pass{Name: x, Type: "kernel", Kernel: gk, Entry: "fft_stage", When: "always",
						Bindings: []graph.Binding{{Name: "src", Resource: name("fft"+x+"_", c)}, {Name: "dst", Resource: name("fft"+x+"2_", c)}},
						Uniforms: []graph.Uniform{b.Uniform(gk, "fft", map[string]any{"n": nc, "stage": ref("s"), "axis": axis})}}, gk, nc, nc))
				}
				b.Pass(graph.Pass{Name: pre + "fft." + label, Type: "repeat", Count: bitsLen(nc), Index: "s", When: "always",
					Swap: [][2]string{{name("fftA_", c), name("fftA2_", c)}, {name("fftB_", c), name("fftB2_", c)}}, Body: body})
			}
		}

		// Assemble into the cascade textures, then mips.
		as := graph.Pass{Name: pre + "assemble", Type: "kernel", Kernel: "ocean_assemble", Entry: "ocean_assemble", When: "always",
			Bindings: []graph.Binding{{Name: "fieldA", Resource: name("fftA_", c)}, {Name: "fieldB", Resource: name("fftB_", c)},
				{Name: "disp", Resource: name("dispTex", c)}, {Name: "slope", Resource: name("slopeTex", c)}},
			Uniforms: []graph.Uniform{b.Uniform("ocean_assemble", "assem", map[string]any{"n": nc, "lambda": ref("lambda")})}}
		b.Pass(place(as, "ocean_assemble", nc, nc))
		for _, tex := range []string{name("dispTex", c), name("slopeTex", c)} {
			mipChain(pre, tex, nc, mips)
		}
		b.Export(name("disp", c), name("dispTex", c)).Export(name("slope", c), name("slopeTex", c))
	}

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
	foamBind := []graph.Binding{{Name: "foamIn", Resource: "foamA"}}
	probeBind := []graph.Binding{}
	for i := 0; i < 4; i++ {
		c := min(i, cascades-1) // an unused cascade slot repeats the last real one
		nc := size(c)
		fu[fmt.Sprintf("n%d", i)], fu[fmt.Sprintf("size%d", i)] = nc, oceanCascades[c].size
		foamBind = append(foamBind, graph.Binding{Name: fmt.Sprintf("fieldB%d", i), Resource: name("fftB_", c)})
		probeBind = append(probeBind, graph.Binding{Name: fmt.Sprintf("disp%d", i), Resource: name("fftA_", c)})
	}
	foamBind = append(foamBind, graph.Binding{Name: "foamOut", Resource: "foamB"}, graph.Binding{Name: "foamTex", Resource: "foamTex"})
	b.Pass(place(graph.Pass{Name: "foam", Type: "kernel", Kernel: "ocean_foam", Entry: "ocean_foam", When: "always",
		Bindings: foamBind, Uniforms: []graph.Uniform{b.Uniform("ocean_foam", "foam", fu)}}, "ocean_foam", foamWidth, foamWidth))
	mipChain("", "foamTex", foamWidth, foamMips)
	b.Swap("foamA", "foamB").Export("foam", "foamTex")

	// Probe (WebGPU only): sample the surface at the app's points for readback.
	if gpu {
		b.FedBuffer("probePoints", "vec4<f32>", probeMax, 0, "probePoints")
		b.Buffer("probeOut", "vec4<f32>", probeMax, 0)
		pu := map[string]any{"count": ref("probeCount"), "cascades": cascades, "t": ref("simTime"), "normalStep": probeStep, "lambda": ref("lambda")}
		for i := 0; i < 4; i++ {
			c := min(i, cascades-1)
			pu[fmt.Sprintf("n%d", i)], pu[fmt.Sprintf("size%d", i)] = size(c), oceanCascades[c].size
		}
		bind := append([]graph.Binding{{Name: "points", Resource: "probePoints"}}, probeBind...)
		bind = append(bind, graph.Binding{Name: "out", Resource: "probeOut"})
		b.Pass(graph.Pass{Name: "probe", Type: "kernel", Kernel: "ocean_probe", Entry: "ocean_probe", When: "always",
			Dispatch: b.Workgroups("ocean_probe", "ocean_probe", probeMax), Bindings: bind,
			Uniforms: []graph.Uniform{b.Uniform("ocean_probe", "probe", pu)}})
		b.Export("probe", "probeOut")
	}
	return b.Build()
}

// bitsLen is log2 of a power of two.
func bitsLen(n int) int { return bits.TrailingZeros(uint(n)) }
