package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
)

// wakeSourceParams mirrors WakeSourceParams. Lengths are metres, speeds m/s,
// angles radians. hullU, hullV give the hull centre relative to the window
// origin corner, in metres.
type wakeSourceParams struct {
	width                int
	reset                int
	originX, originZ     int
	shiftX, shiftZ       int
	cell, dt, tau, depth float64
	hullU, hullV         float64
	heading, speed       float64
	length, beam         float64
	srcLength, srcBeam   float64
	fullSpeed, foamSpeed float64
	diffuse, jitter      float64
}

func (p wakeSourceParams) uniform() map[string]any {
	return map[string]any{
		"width": int64(p.width), "reset": int64(p.reset), "_pad0": int64(0), "_pad1": int64(0),
		"originX": int64(p.originX), "originZ": int64(p.originZ), "shiftX": int64(p.shiftX), "shiftZ": int64(p.shiftZ),
		"cell": p.cell, "dt": p.dt, "tau": p.tau, "depth": p.depth,
		"hullU": p.hullU, "hullV": p.hullV, "heading": p.heading, "speed": p.speed,
		"hullLen": p.length, "hullBeam": p.beam, "srcLength": p.srcLength, "srcBeam": p.srcBeam,
		"fullSpeed": p.fullSpeed, "foamSpeed": p.foamSpeed, "diffuse": p.diffuse, "jitter": p.jitter,
	}
}

func sat(x float64) float64 { return math.Min(math.Max(x, 0), 1) }

// wakeFootprint is the compact smooth ellipse (1 - q)^2, q < 1, with semi axes
// a (along) and b (across).
func wakeFootprint(along, across, a, b float64) float64 {
	q := (along/a)*(along/a) + (across/b)*(across/b)
	if q >= 1 {
		return 0
	}
	return (1 - q) * (1 - q)
}

// wakeHash is the kernel's float hash of a world cell, indices reduced mod 289.
func wakeHash(wx, wz int) float64 {
	x, z := float64(((wx%289)+289)%289), float64(((wz%289)+289)%289)
	v := math.Sin(12.9898*x+78.233*z) * 43758.5453
	return v - math.Floor(v)
}

// wakeSourceRef is the expected output for the cell at window position (ru, rv)
// (texel (ru, rv) too: the tests keep the origin at 0 unless they say so). turbAt
// gives the raw turbulence of a texel of the old state, for the diffusion.
func wakeSourceRef(p wakeSourceParams, ru, rv int, state [4]float64, cleared bool, turbAt func(x, z int) float64) (s float64, clean [4]float64) {
	h, hp, turb, theta := state[0], state[1], state[2], state[3]
	if cleared {
		h, hp, turb, theta = 0, 0, 0, 0
	}
	px, pz := (float64(ru)+0.5)*p.cell-p.hullU, (float64(rv)+0.5)*p.cell-p.hullV
	fx, fz := math.Sin(p.heading), math.Cos(p.heading)
	along, across := px*fx+pz*fz, px*fz-pz*fx
	fade := sat(p.speed / p.fullSpeed)
	bowShift := math.Max(p.length/2-p.srcLength/2, 0)
	s = h + p.depth*fade*fade*wakeFootprint(along-bowShift, across, p.srcLength/2, p.srcBeam/2)
	uh, vh := along/(p.length/2), across/(p.beam/2)
	inj := 0.4 * wakeFootprint(uh, vh, 1, 1)
	{
		su, sv := (uh+0.9)/0.4, vh/1.2
		q := su*su + sv*sv
		if q < 1 {
			x := math.Min(math.Max((1-q)*1.7, 0), 1)
			inj += x * x * (3 - 2*x)
		}
		if qx := 0.4 * q; qx < 1 {
			inj += 0.3 * (1 - qx) * (1 - qx)
		}
	}
	inj += 0.35 * wakeFootprint(uh-0.85, vh, 0.25, 0.45)
	inj = math.Min(inj*sat(p.speed/p.foamSpeed), 1)
	w := p.width
	ka := math.Min(p.diffuse*p.dt/(p.cell*p.cell), 0.2)
	nb := turbAt((ru+w-1)%w, rv) + turbAt((ru+1)%w, rv) + turbAt(ru, (rv+w-1)%w) + turbAt(ru, (rv+1)%w)
	diff := turb + ka*(nb-4*turb)
	decayed := math.Max(diff, 0) * math.Exp(-p.dt*(1+p.jitter*wakeHash(ru+p.originX, rv+p.originZ))/p.tau)
	turbNew := math.Max(decayed, inj)
	if inj > decayed {
		theta = p.heading - math.Pi*math.Floor(p.heading/math.Pi)
	}
	return s, [4]float64{h, hp, turbNew, theta}
}

// runWakeSource runs wake_source over a toroidal window of p.width^2 cells.
func runWakeSource(t *testing.T, p wakeSourceParams, state [][4]float64, opts run.Options) (fftIn, clean [][4]float64) {
	t.Helper()
	m := checkedWakeKernel(t, "wake_source")
	n2 := p.width * p.width
	in, out1, out2 := make([]any, n2), make([]any, n2), make([]any, n2)
	for i := 0; i < n2; i++ {
		v := state[i]
		in[i] = []float64{v[0], v[1], v[2], v[3]}
		out1[i] = []float64{9, 9, 9, 9}
		out2[i] = []float64{9, 9, 9, 9}
	}
	vars := map[string]any{"wake": p.uniform(), "wakeIn": in, "fftOut": out1, "cleanOut": out2}
	if err := run.RunWith(m, "wake_source", n2, &run.Memory{Vars: vars}, opts); err != nil {
		t.Fatal(err)
	}
	conv := func(b []any) [][4]float64 {
		o := make([][4]float64, n2)
		for i := range o {
			v := b[i].([]float64)
			o[i] = [4]float64{v[0], v[1], v[2], v[3]}
		}
		return o
	}
	return conv(out1), conv(out2)
}

func wakeTestParams() wakeSourceParams {
	return wakeSourceParams{width: 32, cell: 4, dt: 0.1, tau: 30, depth: 0.5,
		hullU: 60, hullV: 70, heading: 0.6, speed: 8, length: 40, beam: 12, srcLength: 24, srcBeam: 10, fullSpeed: 4, foamSpeed: 5, diffuse: 1.5, jitter: 1.2}
}

func TestWakeSourceFootprintAndFoam(t *testing.T) {
	p := wakeTestParams()
	n2 := p.width * p.width
	state := make([][4]float64, n2)
	for i := range state {
		state[i] = [4]float64{0.01 * float64(i%7), -0.02 * float64(i%5), 0.3, 0}
	}
	fftIn, clean := runWakeSource(t, p, state, run.Options{})
	var inFoot, injected int
	maxS := 0.0
	for i := 0; i < n2; i++ {
		ru, rv := i%p.width, i/p.width // origin 0: window position equals texel
		s, c := wakeSourceRef(p, ru, rv, state[i], false, func(x, z int) float64 { return state[z*p.width+x][2] })
		if got := fftIn[i]; math.Abs(got[0]-s) > 1e-9 || got[1] != 0 || got[2] != 0 || got[3] != 0 {
			t.Fatalf("cell (%d,%d): fft input %v, want (%g,0,0,0)", ru, rv, got, s)
		}
		for k := 0; k < 4; k++ {
			if math.Abs(clean[i][k]-c[k]) > 1e-9 {
				t.Fatalf("cell (%d,%d): clean %v, want %v", ru, rv, clean[i], c)
			}
		}
		if s != state[i][0] {
			inFoot++
		}
		if c[2] > 0.3+1e-9 {
			injected++
		}
		maxS = math.Max(maxS, s-state[i][0])
	}
	t.Logf("%d cells under the pressure patch (peak %.3f m of %.3f), %d cells with new turbulence", inFoot, maxS, p.depth, injected)
	if inFoot == 0 || injected == 0 {
		t.Fatalf("no footprint or no injection: the test does not exercise the source")
	}
	if maxS > p.depth+1e-9 {
		t.Errorf("pressure exceeds the draft scale: %g > %g", maxS, p.depth)
	}
}

func TestWakeSourceFadesAtLowSpeed(t *testing.T) {
	p := wakeTestParams()
	state := make([][4]float64, p.width*p.width)
	peak := func(speed float64) (pressure, turb float64) {
		p.speed = speed
		fftIn, clean := runWakeSource(t, p, state, run.Options{})
		for i := range fftIn {
			pressure, turb = math.Max(pressure, fftIn[i][0]), math.Max(turb, clean[i][2])
		}
		return
	}
	p0, t0 := peak(0)
	pHalf, _ := peak(2)
	pFull, tFull := peak(8)
	if p0 != 0 || t0 != 0 {
		t.Errorf("a hull at rest forces (%g) or foams (%g)", p0, t0)
	}
	// Half the full speed gives a quarter of the pressure (fade squared).
	if math.Abs(pHalf-0.25*pFull) > 1e-12 || pFull <= 0 || tFull <= 0 {
		t.Errorf("pressure at half the full speed %g, want a quarter of %g", pHalf, pFull)
	}
}

// A shift moves the window origin by whole cells; cells that stay keep their
// buffer slot (so the field is preserved in world terms) and cells that wrap in
// are cleared. reset clears everything.
func TestWakeSourceWindowShift(t *testing.T) {
	const w = 16
	for _, sh := range []struct{ dx, dz int }{{3, -2}, {-5, 4}, {0, 0}, {w, 1}} {
		p := wakeTestParams()
		p.width, p.speed, p.depth = w, 0, 0
		p.originX, p.originZ, p.shiftX, p.shiftZ = 20, -5, sh.dx, sh.dz
		state := make([][4]float64, w*w)
		for i := range state {
			state[i] = [4]float64{1 + float64(i), 2 + float64(i), 3, 0}
		}
		_, clean := runWakeSource(t, p, state, run.Options{})
		cleared := 0
		for v := 0; v < w; v++ {
			for u := 0; u < w; u++ {
				ru, rv := posMod(u-p.originX, w), posMod(v-p.originZ, w)
				entering := (sh.dx > 0 && ru >= w-sh.dx) || (sh.dx < 0 && ru < -sh.dx) || (sh.dz > 0 && rv >= w-sh.dz) || (sh.dz < 0 && rv < -sh.dz)
				want := state[v*w+u]
				if entering {
					want = [4]float64{}
					cleared++
				} else {
					want[2] *= math.Exp(-p.dt * (1 + p.jitter*wakeHash(ru+p.originX, rv+p.originZ)) / p.tau)
				}
				got := clean[v*w+u]
				for k := 0; k < 3; k++ {
					if math.Abs(got[k]-want[k]) > 1e-9 {
						t.Fatalf("shift (%d,%d) texel (%d,%d): %v, want %v", sh.dx, sh.dz, u, v, got, want)
					}
				}
			}
		}
		t.Logf("shift (%d,%d): %d of %d cells cleared, the rest kept", sh.dx, sh.dz, cleared, w*w)
	}
	p := wakeTestParams()
	p.width, p.reset, p.speed = w, 1, 0
	state := make([][4]float64, w*w)
	for i := range state {
		state[i] = [4]float64{1, 1, 1, 0}
	}
	_, clean := runWakeSource(t, p, state, run.Options{})
	for i, c := range clean {
		if c != [4]float64{} {
			t.Fatalf("reset: cell %d = %v, want 0", i, c)
		}
	}
}

// peakAlong returns the largest and smallest `along` (m, from the hull centre) of the cells where f holds.
func extentAlong(p wakeSourceParams, vals [][4]float64, lane int, over float64) (lo, hi float64) {
	lo, hi = math.Inf(1), math.Inf(-1)
	fx, fz := math.Sin(p.heading), math.Cos(p.heading)
	for i, v := range vals {
		if v[lane] <= over {
			continue
		}
		px, pz := (float64(i%p.width)+0.5)*p.cell-p.hullU, (float64(i/p.width)+0.5)*p.cell-p.hullV
		a := px*fx + pz*fz
		lo, hi = math.Min(lo, a), math.Max(hi, a)
	}
	return
}

// The pressure patch (and with it the Kelvin pattern) starts at the bow: its forward edge is the hull's bow, not its middle.
func TestWakeSourcePressureStartsAtTheBow(t *testing.T) {
	p := wakeTestParams()
	p.width, p.cell, p.hullU, p.hullV, p.heading = 64, 2, 64, 64, 0
	p.length, p.srcLength, p.beam, p.srcBeam = 60, 20, 12, 8
	fftIn, _ := runWakeSource(t, p, make([][4]float64, p.width*p.width), run.Options{})
	lo, hi := extentAlong(p, fftIn, 0, 1e-6)
	if hi > p.length/2+p.cell || hi < p.length/2-2*p.cell {
		t.Errorf("pressure reaches %.1f m ahead of the hull centre, want about the bow at %.1f", hi, p.length/2)
	}
	if lo < p.length/2-p.srcLength-p.cell {
		t.Errorf("pressure reaches back to %.1f m, want no further than %.1f", lo, p.length/2-p.srcLength)
	}
}

// The churned band is about one beam wide, starts at the stern, and the streak angle is the heading folded into [0, pi).
func TestWakeSourceSternWashAndStreakAngle(t *testing.T) {
	p := wakeTestParams()
	p.width, p.cell, p.hullU, p.hullV, p.heading = 64, 2, 64, 64, 0.6+math.Pi
	p.length, p.beam, p.speed, p.diffuse, p.jitter = 60, 12, 8, 0, 0
	_, clean := runWakeSource(t, p, make([][4]float64, p.width*p.width), run.Options{})
	fx, fz := math.Sin(p.heading), math.Cos(p.heading)
	var wideLo, wideHi = math.Inf(1), math.Inf(-1)
	for i, v := range clean {
		if v[2] < 0.9 {
			continue
		}
		px, pz := (float64(i%p.width)+0.5)*p.cell-p.hullU, (float64(i/p.width)+0.5)*p.cell-p.hullV
		across := px*fz - pz*fx
		wideLo, wideHi = math.Min(wideLo, across), math.Max(wideHi, across)
		if want := p.heading - math.Pi*math.Floor(p.heading/math.Pi); math.Abs(v[3]-want) > 1e-9 {
			t.Fatalf("streak angle %g, want %g", v[3], want)
		}
	}
	if w := wideHi - wideLo; w < 0.5*p.beam || w > 1.2*p.beam {
		t.Errorf("fully churned band is %.1f m wide, want about one beam (%.0f m)", w, p.beam)
	}
	_, hi := extentAlong(p, clean, 2, 0.9)
	if hi > 0 {
		t.Errorf("the fully churned water reaches %.1f m ahead of the hull centre, want it behind (the stern wash)", hi)
	}
}

// Diffusion widens a stripe of turbulence without creating any; the decay jitter makes the cells fade at different rates.
func TestWakeSourceDiffusionAndJitter(t *testing.T) {
	p := wakeTestParams()
	p.width, p.speed, p.depth = 32, 0, 0
	p.diffuse, p.jitter = 8, 0
	state := make([][4]float64, p.width*p.width)
	for i := range state {
		if i%p.width == 16 {
			state[i][2] = 1
		}
	}
	_, clean := runWakeSource(t, p, state, run.Options{})
	left, centre := clean[5*p.width+15][2], clean[5*p.width+16][2]
	if left <= 0 || centre >= math.Exp(-p.dt/p.tau) {
		t.Errorf("a stripe should spread: left neighbour %g, centre %g", left, centre)
	}
	sumBefore, sumAfter := 0.0, 0.0
	for i := range state {
		sumBefore += state[i][2]
		sumAfter += clean[i][2]
	}
	if sumAfter > sumBefore*math.Exp(-p.dt/p.tau)+1e-9 {
		t.Errorf("diffusion created turbulence: %g after, %g before", sumAfter, sumBefore)
	}
	p.diffuse, p.jitter = 0, 2
	for i := range state {
		state[i][2] = 1
	}
	_, clean = runWakeSource(t, p, state, run.Options{})
	lo, hi := 1.0, 0.0
	for _, c := range clean {
		lo, hi = math.Min(lo, c[2]), math.Max(hi, c[2])
	}
	if hi-lo < 0.05*(1-math.Exp(-p.dt*3/p.tau)) {
		t.Errorf("with jitter the cells still fade alike: %g..%g", lo, hi)
	}
}
