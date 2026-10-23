package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
)

// wakeStepParams mirrors WakeStepParams.
type wakeStepParams struct {
	width            int
	originX, originZ int
	sponge           float64 // cells
	spongeRate       float64 // 1/s at the window edge
	dt, gravity      float64
	alpha, cell      float64
}

func (p wakeStepParams) uniform() map[string]any {
	return map[string]any{
		"width": int64(p.width), "_pad0": int64(0), "_pad1": int64(0), "_pad2": int64(0),
		"originX": int64(p.originX), "originZ": int64(p.originZ), "sponge": p.sponge, "spongeRate": p.spongeRate,
		"dt": p.dt, "gravity": p.gravity, "alpha": p.alpha, "cell": p.cell,
	}
}

func vec4Buf(v [][4]float64) []any {
	b := make([]any, len(v))
	for i, e := range v {
		b[i] = []float64{e[0], e[1], e[2], e[3]}
	}
	return b
}

// runWakeStep runs wake_step with q holding Q[s] in its x lane and clean the
// cleaned state; it returns the new state and the output texture.
func runWakeStep(t *testing.T, p wakeStepParams, q, clean [][4]float64, opts run.Options) ([][4]float64, *run.Texture) {
	t.Helper()
	m := checkedWakeKernel(t, "wake_step")
	n2 := p.width * p.width
	out := make([]any, n2)
	for i := range out {
		out[i] = []float64{9, 9, 9, 9}
	}
	tex := &run.Texture{Width: p.width, Height: p.width, Format: "rgba16float", Data: make([]float64, n2*4)}
	vars := map[string]any{"step": p.uniform(), "qIn": vec4Buf(q), "cleanIn": vec4Buf(clean), "stateOut": out, "wakeTex": tex}
	if err := run.RunWith(m, "wake_step", n2, &run.Memory{Vars: vars}, opts); err != nil {
		t.Fatal(err)
	}
	st := make([][4]float64, n2)
	for i := range st {
		v := out[i].([]float64)
		st[i] = [4]float64{v[0], v[1], v[2], v[3]}
	}
	return st, tex
}

// wakeStepRef is the leapfrog update with sponge damping at window position
// (ru, rv) of a width^2 window.
func wakeStepRef(p wakeStepParams, c, q [4]float64, ru, rv int) float64 {
	e := min(min(ru, rv), min(p.width-1-ru, p.width-1-rv))
	tt := sat((p.sponge - float64(e)) / math.Max(p.sponge, 0.001))
	beta := math.Max(1-(p.alpha+p.spongeRate*tt*tt)*p.dt, 0)
	return c[0] + beta*(c[0]-c[1]) - p.dt*p.dt*p.gravity*q[0]
}

func TestWakeStepMatchesReference(t *testing.T) {
	const w = 16
	p := wakeStepParams{width: w, originX: 21, originZ: -6, sponge: 4, spongeRate: 0.5, dt: 0.1, gravity: 9.81, alpha: 0.02, cell: 4}
	q, clean := make([][4]float64, w*w), make([][4]float64, w*w)
	for i := range q {
		q[i] = [4]float64{0.05 * math.Sin(float64(i)*0.37), 0, 0, 0}
		clean[i] = [4]float64{0.3 * math.Cos(float64(i)*0.11), 0.28 * math.Cos(float64(i)*0.11+0.2), 0.5 + 0.01*float64(i%9), 0}
	}
	state, tex := runWakeStep(t, p, q, clean, run.Options{})
	hn := func(u, v int) float64 { // new height at buffer texel (u, v), wrapped
		u, v = posMod(u, w), posMod(v, w)
		return wakeStepRef(p, clean[v*w+u], q[v*w+u], posMod(u-p.originX, w), posMod(v-p.originZ, w))
	}
	var damped int
	for v := 0; v < w; v++ {
		for u := 0; u < w; u++ {
			i := v*w + u
			want := hn(u, v)
			if math.Abs(state[i][0]-want) > 1e-12 || state[i][1] != clean[i][0] || state[i][2] != clean[i][2] || state[i][3] != 0 {
				t.Fatalf("texel (%d,%d): state %v, want (%g, %g, %g, 0)", u, v, state[i], want, clean[i][0], clean[i][2])
			}
			dhdx := (hn(u+1, v) - hn(u-1, v)) / (2 * p.cell)
			dhdz := (hn(u, v+1) - hn(u, v-1)) / (2 * p.cell)
			tx := tex.Data[4*i:]
			for k, w := range [4]float64{want, dhdx, dhdz, clean[i][2]} {
				if tx[k] != roundHalf(w) {
					t.Fatalf("texel (%d,%d): texture lane %d = %g, want %g", u, v, k, tx[k], roundHalf(w))
				}
			}
			if ru := posMod(u-p.originX, w); ru < 4 {
				damped++
			}
		}
	}
	t.Logf("%d texels checked, %d in the sponge band on the low x edge", w*w, damped)
}

func TestWakeStepPureLeapfrog(t *testing.T) {
	// No operator output, no damping: h' = 2h - h_prev.
	const w = 8
	p := wakeStepParams{width: w, dt: 0.05, gravity: 9.81, cell: 2}
	q, clean := make([][4]float64, w*w), make([][4]float64, w*w)
	for i := range clean {
		clean[i] = [4]float64{1, 0.25, 0, 0}
	}
	state, _ := runWakeStep(t, p, q, clean, run.Options{})
	for i, s := range state {
		if s[0] != 1.75 {
			t.Fatalf("cell %d: h' = %g, want 1.75", i, s[0])
		}
	}
}
