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
}

func (p wakeSourceParams) uniform() map[string]any {
	return map[string]any{
		"width": int64(p.width), "reset": int64(p.reset), "_pad0": int64(0), "_pad1": int64(0),
		"originX": int64(p.originX), "originZ": int64(p.originZ), "shiftX": int64(p.shiftX), "shiftZ": int64(p.shiftZ),
		"cell": p.cell, "dt": p.dt, "tau": p.tau, "depth": p.depth,
		"hullU": p.hullU, "hullV": p.hullV, "heading": p.heading, "speed": p.speed,
		"hullLen": p.length, "hullBeam": p.beam, "srcLength": p.srcLength, "srcBeam": p.srcBeam,
		"fullSpeed": p.fullSpeed, "foamSpeed": p.foamSpeed, "_pad2": 0.0, "_pad3": 0.0,
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

// wakeSourceRef is the expected output for the cell at window position (ru, rv).
func wakeSourceRef(p wakeSourceParams, ru, rv int, state [4]float64, cleared bool) (s float64, clean [4]float64) {
	h, hp, turb := state[0], state[1], state[2]
	if cleared {
		h, hp, turb = 0, 0, 0
	}
	px, pz := (float64(ru)+0.5)*p.cell-p.hullU, (float64(rv)+0.5)*p.cell-p.hullV
	fx, fz := math.Sin(p.heading), math.Cos(p.heading)
	along, across := px*fx+pz*fz, px*fz-pz*fx
	fade := sat(p.speed / p.fullSpeed)
	s = h + p.depth*fade*fade*wakeFootprint(along, across, p.srcLength/2, p.srcBeam/2)
	uh := along / (p.length / 2)
	wh := wakeFootprint(along, across, p.length/2, p.beam/2)
	bowU, bowV := (uh-0.85)/0.25, across/(p.beam/2)/0.45
	wb := 0.0
	if q := bowU*bowU + bowV*bowV; q < 1 {
		wb = (1 - q) * (1 - q)
	}
	inj := sat(p.speed/p.foamSpeed) * (0.8*wh*(0.5-0.5*math.Min(math.Max(uh, -1), 1)) + 0.6*wb)
	turbNew := math.Max(turb*math.Exp(-p.dt/p.tau), math.Min(inj, 1))
	return s, [4]float64{h, hp, turbNew, 0}
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
		hullU: 60, hullV: 70, heading: 0.6, speed: 8, length: 40, beam: 12, srcLength: 24, srcBeam: 10, fullSpeed: 4, foamSpeed: 5}
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
		s, c := wakeSourceRef(p, ru, rv, state[i], false)
		if got := fftIn[i]; math.Abs(got[0]-s) > 1e-12 || got[1] != 0 || got[2] != 0 || got[3] != 0 {
			t.Fatalf("cell (%d,%d): fft input %v, want (%g,0,0,0)", ru, rv, got, s)
		}
		for k := 0; k < 4; k++ {
			if math.Abs(clean[i][k]-c[k]) > 1e-12 {
				t.Fatalf("cell (%d,%d): clean %v, want %v", ru, rv, clean[i], c)
			}
		}
		if s != state[i][0] {
			inFoot++
		}
		if c[2] > 0.3*math.Exp(-p.dt/p.tau)+1e-9 {
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
					want[2] *= math.Exp(-p.dt / p.tau)
				}
				got := clean[v*w+u]
				for k := 0; k < 3; k++ {
					if math.Abs(got[k]-want[k]) > 1e-12 {
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
