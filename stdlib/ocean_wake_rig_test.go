package stdlib

import (
	"math"
	"math/cmplx"
	"sync"
	"testing"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// fftInverseGo is the unnormalized inverse 2D FFT (e^{+i}) of the Elio kernels
// in plain Go, for tests that run hundreds of steps. TestWakeRigFFTAgrees checks
// it against the Elio gather kernel.
func fftInverseGo(n int, buf []any) []any {
	out := make([]any, n*n)
	slots := [2][]complex128{make([]complex128, n*n), make([]complex128, n*n)}
	for i := range buf {
		f := buf[i].([]float64)
		slots[0][i], slots[1][i] = complex(f[0], f[1]), complex(f[2], f[3])
	}
	line := make([]complex128, n)
	for _, a := range slots {
		for z := 0; z < n; z++ {
			copy(line, a[z*n:(z+1)*n])
			fftLine(line)
			copy(a[z*n:(z+1)*n], line)
		}
		for x := 0; x < n; x++ {
			for z := 0; z < n; z++ {
				line[z] = a[z*n+x]
			}
			fftLine(line)
			for z := 0; z < n; z++ {
				a[z*n+x] = line[z]
			}
		}
	}
	for i := range out {
		out[i] = []float64{real(slots[0][i]), imag(slots[0][i]), real(slots[1][i]), imag(slots[1][i])}
	}
	return out
}

// fftLine is an in-place radix-2 inverse transform (positive exponent).
func fftLine(a []complex128) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for l := 2; l <= n; l <<= 1 {
		wl := cmplx.Rect(1, 2*math.Pi/float64(l))
		for i := 0; i < n; i += l {
			w := complex(1, 0)
			for j := 0; j < l/2; j++ {
				u, v := a[i+j], a[i+j+l/2]*w
				a[i+j], a[i+j+l/2] = u+v, u-v
				w *= wl
			}
		}
	}
}

func TestWakeRigFFTAgrees(t *testing.T) {
	const n = 16
	spec := randomSpectrum(n, 7)
	// The gather passes ping-pong through their input buffer, so give each a copy.
	b := unpackVec4(fftInverseGo(n, packVec4(spec)))
	a := unpackVec4(fftInverseGather(t, n, packVec4(spec)))
	if e := maxErr(a, b); e > 1e-9 {
		t.Fatalf("Go FFT differs from the Elio gather FFT by %g", e)
	}
}

var (
	wakeKernelMu    sync.Mutex
	wakeKernelCache = map[string]*ir.Module{}
)

// checkedWakeKernel builds a wake kernel once per test binary and checks it
// with sema.Check and sema.CheckGather (the interpreter never changes it).
func checkedWakeKernel(t *testing.T, name string) *ir.Module {
	t.Helper()
	wakeKernelMu.Lock()
	defer wakeKernelMu.Unlock()
	if m, ok := wakeKernelCache[name]; ok {
		return m
	}
	build := map[string]func() (*ir.Module, error){"wake_source": WakeSource, "wake_symbol": WakeSymbol, "wake_step": WakeStep}[name]
	m, err := build()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	if errs := sema.CheckGather(m, name); len(errs) != 0 {
		t.Fatalf("CheckGather: %v", errs)
	}
	wakeKernelCache[name] = m
	return m
}

// hullState is the hull in window coordinates, metres from the window origin
// corner.
type hullState struct {
	u, v           float64
	heading, speed float64
	depth          float64
}

// wakeRig runs the four wake kernels as the graph orders them each step.
type wakeRig struct {
	t            *testing.T
	n            int
	cell         float64
	gravity, dt  float64
	alpha        float64
	sponge, rate float64
	tau          float64
	elioFFT      bool // use the Elio gather FFT; otherwise the Go one
	srcLength    float64
	srcBeam      float64
	hullLength   float64
	hullBeam     float64
	originX      int
	originZ      int
	state        [][4]float64
	tex          *run.Texture
}

func newWakeRig(t *testing.T, n int, cell float64) *wakeRig {
	return &wakeRig{t: t, n: n, cell: cell, gravity: 9.81, dt: 0.05, tau: 30, srcLength: 6 * cell, srcBeam: 4 * cell,
		hullLength: 10 * cell, hullBeam: 4 * cell, state: make([][4]float64, n*n)}
}

// step advances the wake one step with the hull h and a window move of
// (shiftX, shiftZ) whole cells (the rig's origin moves with it).
func (r *wakeRig) step(h hullState, shiftX, shiftZ int) {
	t, n := r.t, r.n
	r.originX += shiftX
	r.originZ += shiftZ
	sp := wakeSourceParams{width: n, originX: r.originX, originZ: r.originZ, shiftX: shiftX, shiftZ: shiftZ,
		cell: r.cell, dt: r.dt, tau: r.tau, depth: h.depth, hullU: h.u, hullV: h.v, heading: h.heading, speed: h.speed,
		length: r.hullLength, beam: r.hullBeam, srcLength: r.srcLength, srcBeam: r.srcBeam, fullSpeed: 1e-9, foamSpeed: 5}
	fftIn, clean := runWakeSource(t, sp, r.state, run.Options{})
	fft := fftInverseGo
	if r.elioFFT {
		fft = func(n int, b []any) []any { return fftInverseGather(t, n, b) }
	}
	u := fft(n, vec4Buf(fftIn))
	g := runWakeSymbol(t, n, float64(n)*r.cell, u)
	q := fft(n, g)
	qs := make([][4]float64, n*n)
	for i := range qs {
		qs[i][0] = q[i].([]float64)[0]
	}
	st := wakeStepParams{width: n, originX: r.originX, originZ: r.originZ, sponge: r.sponge, spongeRate: r.rate,
		dt: r.dt, gravity: r.gravity, alpha: r.alpha, cell: r.cell}
	r.state, r.tex = runWakeStep(t, st, qs, clean, run.Options{})
}

// window returns h in window order: window[rv*n+ru] for window cell (ru, rv).
func (r *wakeRig) window() []float64 {
	n := r.n
	w := make([]float64, n*n)
	for v := 0; v < n; v++ {
		for u := 0; u < n; u++ {
			w[v*n+u] = r.state[posMod(v+r.originZ, n)*n+posMod(u+r.originX, n)][0]
		}
	}
	return w
}
