package stdlib

import (
	"math"
	"math/bits"
	"math/cmplx"
	"math/rand"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// fftGather2D runs the inverse 2D FFT in its gather form over buf (n*n vec4).
func fftGather2D(t *testing.T, n int, buf []any) []any {
	return fftGather2DWith(t, n, buf, run.Options{})
}

func fftGather2DWith(t *testing.T, n int, buf []any, opts run.Options) []any {
	t.Helper()
	m, err := FFT2DGather(n)
	if err != nil {
		t.Fatal(err)
	}
	src, dst := buf, make([]any, n*n)
	for i := range dst {
		dst[i] = []float64{0, 0, 0, 0}
	}
	log2n := bits.TrailingZeros(uint(n))
	for axis := int64(0); axis < 2; axis++ {
		for stage := 0; stage < log2n; stage++ {
			mem := &run.Memory{Vars: map[string]any{
				"fft": map[string]any{"n": int64(n), "stage": int64(stage), "axis": axis, "_pad": int64(0)},
				"src": src, "dst": dst,
			}}
			if err := run.RunWith(m, "fft_stage", n*n, mem, opts); err != nil {
				t.Fatal(err)
			}
			src, dst = dst, src
		}
	}
	return src
}

// evolveField names the eight real fields in packing order.
var evolveField = [8]string{"Dy", "Dx", "Dz", "dDy/dx", "dDy/dz", "dDx/dx", "dDz/dz", "dDx/dz"}

// runOceanEvolve evolves h0 to time t and returns the eight real fields after
// the inverse FFT: fields[f][z*n+x], in the order of evolveField.
func runOceanEvolve(t *testing.T, s oceanSpec, h0 [][]float64, omega []float64, tm float64) [8][]float64 {
	t.Helper()
	return runOceanEvolveWith(t, s, h0, omega, tm, run.Options{})
}

// runOceanEvolveWith is runOceanEvolve with interpreter options (Float32 mode
// applies to the evolve kernel and to the FFT passes).
func runOceanEvolveWith(t *testing.T, s oceanSpec, h0 [][]float64, omega []float64, tm float64, opts run.Options) [8][]float64 {
	t.Helper()
	m, err := OceanEvolve()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	n := s.n
	in := make([]any, n*n)
	for i := range in {
		in[i] = h0[i]
	}
	outA, outB := make([]any, n*n), make([]any, n*n)
	for i := range outA {
		outA[i] = []float64{9, 9, 9, 9}
		outB[i] = []float64{9, 9, 9, 9}
	}
	mem := &run.Memory{Vars: map[string]any{
		"ev": map[string]any{"n": int64(n), "_pad0": int64(0), "_pad1": int64(0), "_pad2": int64(0),
			"t": tm, "tile": s.tile, "loop": s.loop, "_pad4": 0.0},
		"h0": in, "omega": omega, "outA": outA, "outB": outB,
	}}
	if err := run.RunWith(m, "ocean_evolve", n*n, mem, opts); err != nil {
		t.Fatal(err)
	}
	a, b := fftGather2DWith(t, n, outA, opts), fftGather2DWith(t, n, outB, opts)
	var f [8][]float64
	for k := range f {
		f[k] = make([]float64, n*n)
	}
	for i := 0; i < n*n; i++ {
		va, vb := a[i].([]float64), b[i].([]float64)
		for c := 0; c < 4; c++ {
			f[c][i] = va[c]
			f[4+c][i] = vb[c]
		}
	}
	return f
}

// directEvolve evaluates the eight fields at continuous position (px, pz) by a
// direct sum over every bin, from h0 and omega, in float64.
func directEvolve(s oceanSpec, h0 [][]float64, omega []float64, tm, px, pz float64) [8]float64 {
	n := s.n
	dk := 2 * math.Pi / s.tile
	var out [8]float64
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			g := h0[z*n+x]
			if g[0] == 0 && g[1] == 0 && g[2] == 0 && g[3] == 0 {
				continue
			}
			ix, iz := signedBin(x, n), signedBin(z, n)
			kx, kz := dk*float64(ix), dk*float64(iz)
			kk := math.Hypot(kx, kz)
			w := omega[z*n+x]
			h := complex(g[0], g[1])*cmplx.Exp(complex(0, w*tm)) + complex(g[2], g[3])*cmplx.Exp(complex(0, -w*tm))
			e := cmplx.Exp(complex(0, kx*px+kz*pz))
			he := h * e
			ux, uz := kx/kk, kz/kk
			out[0] += real(he)
			out[1] += real(complex(0, ux) * he)
			out[2] += real(complex(0, uz) * he)
			out[3] += real(complex(0, kx) * he)
			out[4] += real(complex(0, kz) * he)
			out[5] += real(complex(-kx*ux, 0) * he)
			out[6] += real(complex(-kz*uz, 0) * he)
			out[7] += real(complex(-kx*uz, 0) * he)
		}
	}
	return out
}

func rmsReal(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s / float64(len(v)))
}

func TestOceanEvolveMatchesDirectSum(t *testing.T) {
	s := defaultSpec(1, 64)
	s.seed = 5
	h0, omega := runOceanSpectrum(t, s)
	const tm = 3.7
	f := runOceanEvolve(t, s, h0, omega, tm)
	n := s.n
	rms := [8]float64{}
	for k := range f {
		rms[k] = rmsReal(f[k])
		if rms[k] == 0 {
			t.Fatalf("field %s is identically zero", evolveField[k])
		}
	}
	r := rand.New(rand.NewSource(11))
	var worst, worstDy float64
	for p := 0; p < 16; p++ {
		x, z := r.Intn(n), r.Intn(n)
		d := directEvolve(s, h0, omega, tm, float64(x)*s.tile/float64(n), float64(z)*s.tile/float64(n))
		for k := 0; k < 8; k++ {
			e := math.Abs(f[k][z*n+x]-d[k]) / rms[k]
			if e > worst {
				worst = e
			}
			if k == 0 && e > worstDy {
				worstDy = e
			}
			if e > 1e-9 {
				t.Errorf("sample (%d,%d) field %s: fft %.12g direct %.12g (error %.3g of RMS)", x, z, evolveField[k], f[k][z*n+x], d[k], e)
			}
		}
	}
	t.Logf("evolve vs direct sum, 16 samples x 8 fields, n=%d: max error %.3g of field RMS (Dy alone %.3g)", n, worst, worstDy)
}

// TestOceanEvolveFieldsAreConsistent checks the derivative fields against
// finite differences of the displacement fields, the sign of the horizontal
// displacement, and the absence of leakage between the two fields packed in
// one complex number (all values at sample points, from the direct sum).
func TestOceanEvolveFieldsAreConsistent(t *testing.T) {
	s := defaultSpec(2, 64)
	s.seed = 21
	h0, omega := runOceanSpectrum(t, s)
	const tm = 1.25
	f := runOceanEvolve(t, s, h0, omega, tm)
	n := s.n
	// Leakage: each packed real field matches its own direct sum, so neither
	// bleeds into the other; the tolerance is far below the field RMS.
	var maxLeak float64
	r := rand.New(rand.NewSource(3))
	for p := 0; p < 8; p++ {
		x, z := r.Intn(n), r.Intn(n)
		px, pz := float64(x)*s.tile/float64(n), float64(z)*s.tile/float64(n)
		d := directEvolve(s, h0, omega, tm, px, pz)
		for k := 0; k < 8; k++ {
			if e := math.Abs(f[k][z*n+x]-d[k]) / rmsReal(f[k]); e > maxLeak {
				maxLeak = e
			}
		}
		// Finite differences of the direct sums at the same point.
		const eps = 1e-3
		dxp := directEvolve(s, h0, omega, tm, px+eps, pz)
		dxm := directEvolve(s, h0, omega, tm, px-eps, pz)
		dzp := directEvolve(s, h0, omega, tm, px, pz+eps)
		dzm := directEvolve(s, h0, omega, tm, px, pz-eps)
		fd := [8]float64{3: (dxp[0] - dxm[0]) / (2 * eps), 4: (dzp[0] - dzm[0]) / (2 * eps),
			5: (dxp[1] - dxm[1]) / (2 * eps), 6: (dzp[2] - dzm[2]) / (2 * eps), 7: (dzp[1] - dzm[1]) / (2 * eps)}
		for _, k := range []int{3, 4, 5, 6, 7} {
			if e := math.Abs(f[k][z*n+x]-fd[k]) / rmsReal(f[k]); e > 1e-4 {
				t.Errorf("field %s at (%d,%d): %g, finite difference %g", evolveField[k], x, z, f[k][z*n+x], fd[k])
			}
		}
		// d/dx of Dz equals d/dz of Dx.
		if e := math.Abs((dxp[2]-dxm[2])/(2*eps) - fd[7]); e > 1e-4*rmsReal(f[7]) {
			t.Errorf("(%d,%d): dDz/dx %g != dDx/dz %g", x, z, (dxp[2]-dxm[2])/(2*eps), fd[7])
		}
	}
	t.Logf("packed-field leakage: max %.3g of RMS", maxLeak)
	if maxLeak > 1e-12 {
		t.Errorf("leakage %.3g of RMS exceeds 1e-12", maxLeak)
	}
	// Displacement points toward crests for a positive choppiness: Dx and the
	// slope dDy/dx are positively correlated (Parseval: sum of kx^2/|k| |h|^2 > 0),
	// and so are Dz and dDy/dz.
	var cx, cz float64
	for i := 0; i < n*n; i++ {
		cx += f[1][i] * f[3][i]
		cz += f[2][i] * f[4][i]
	}
	if cx <= 0 || cz <= 0 {
		t.Errorf("displacement does not point toward crests: sum Dx*dDy/dx = %g, sum Dz*dDy/dz = %g", cx, cz)
	}
}

// TestOceanEvolveLoops checks the quantized dispersion makes the field exactly
// periodic in T_loop, and that float32 evaluation at a large time, reduced
// modulo T_loop by the host, stays close to the float64 field.
func TestOceanEvolveLoops(t *testing.T) {
	s := defaultSpec(1, 64)
	s.seed = 8
	h0, omega := runOceanSpectrum(t, s)
	a := runOceanEvolve(t, s, h0, omega, 7.3)
	b := runOceanEvolve(t, s, h0, omega, 7.3+s.loopPeriod())
	var worst float64
	for k := 0; k < 8; k++ {
		rms := rmsReal(a[k])
		for i := range a[k] {
			worst = math.Max(worst, math.Abs(a[k][i]-b[k][i])/rms)
		}
	}
	t.Logf("field at t and t+T_loop (float64): max difference %.3g of RMS", worst)
	if worst > 1e-9 {
		t.Errorf("field at t and t + T_loop differs by %.3g of RMS", worst)
	}

	const big = 3600.25
	want := runOceanEvolve(t, s, h0, omega, big) // float64 at the full time
	got := runOceanEvolveWith(t, s, h0, omega, math.Mod(big, s.loopPeriod()), run.Options{Float32: true})
	worst = 0
	for k := 0; k < 8; k++ {
		rms := rmsReal(want[k])
		for i := range want[k] {
			worst = math.Max(worst, math.Abs(want[k][i]-got[k][i])/rms)
		}
	}
	t.Logf("float32 at t = %v (host passes t mod T_loop = %v) vs float64 at t: max error %.3g of RMS", big, math.Mod(big, s.loopPeriod()), worst)
	if worst > 1e-4 {
		t.Errorf("float32 field error %.3g of RMS exceeds 1e-4", worst)
	}
}
