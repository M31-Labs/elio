package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// fieldBuffers packs the eight real fields of runOceanEvolve into the two vec4
// buffers the inverse FFT leaves behind: A = (Dy, Dx, Dz, dDy/dx) and
// B = (dDy/dz, dDx/dx, dDz/dz, dDx/dz).
func fieldBuffers(f [8][]float64) (a, b []any) {
	n2 := len(f[0])
	a, b = make([]any, n2), make([]any, n2)
	for i := 0; i < n2; i++ {
		a[i] = []float64{f[0][i], f[1][i], f[2][i], f[3][i]}
		b[i] = []float64{f[4][i], f[5][i], f[6][i], f[7][i]}
	}
	return a, b
}

// runOceanAssemble runs ocean_assemble and returns the disp and slope textures.
func runOceanAssemble(t *testing.T, n int, a, b []any, lambda float64, opts run.Options) (disp, slope *run.Texture) {
	t.Helper()
	m, err := OceanAssemble()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	if errs := sema.CheckGather(m, "ocean_assemble"); len(errs) != 0 {
		t.Fatalf("CheckGather: %v", errs)
	}
	disp = &run.Texture{Width: n, Height: n, Format: "rgba16float", Data: make([]float64, n*n*4)}
	slope = &run.Texture{Width: n, Height: n, Format: "rgba16float", Data: make([]float64, n*n*4)}
	mem := &run.Memory{Vars: map[string]any{
		"assem":    map[string]any{"n": int64(n), "_pad0": int64(0), "_pad1": int64(0), "_pad2": int64(0), "lambda": lambda, "_pad3": 0.0, "_pad4": 0.0, "_pad5": 0.0},
		"fieldA": a, "fieldB": b, "disp": disp, "slope": slope,
	}}
	if err := run.RunWith(m, "ocean_assemble", n*n, mem, opts); err != nil {
		t.Fatal(err)
	}
	return disp, slope
}

// assembled returns the ideal (unrounded) disp and slope values for texel i.
func assembled(a, b []any, i int, lambda float64) (disp, slope [4]float64) {
	va, vb := a[i].([]float64), b[i].([]float64)
	disp = [4]float64{lambda * va[1], va[0], lambda * va[2], lambda * vb[3]}
	slope = [4]float64{va[3], vb[0], lambda * vb[1], lambda * vb[2]}
	return
}

func TestOceanAssembleZeroChoppiness(t *testing.T) {
	s := defaultSpec(1, 64)
	s.seed = 4
	h0, omega := runOceanSpectrum(t, s)
	a, b := fieldBuffers(runOceanEvolve(t, s, h0, omega, 2))
	n := s.n
	disp, slope := runOceanAssemble(t, n, a, b, 0, run.Options{})
	for i := 0; i < n*n; i++ {
		d, sl := disp.Data[4*i:4*i+4], slope.Data[4*i:4*i+4]
		if d[0] != 0 || d[2] != 0 || d[3] != 0 {
			t.Fatalf("texel %d: with lambda = 0, (Dx, Dz, dDx/dz) = (%g, %g, %g), want 0", i, d[0], d[2], d[3])
		}
		if sl[2] != 0 || sl[3] != 0 {
			t.Fatalf("texel %d: with lambda = 0, (dDx/dx, dDz/dz) = (%g, %g), want 0", i, sl[2], sl[3])
		}
		// The combined Jacobian (1 + dDx/dx)(1 + dDz/dz) - (dDx/dz)^2 is 1.
		if j := (1+sl[2])*(1+sl[3]) - d[3]*d[3]; j != 1 {
			t.Fatalf("texel %d: Jacobian %g, want 1", i, j)
		}
		if want := roundHalf(vec4At(a, i)[0]); d[1] != want {
			t.Fatalf("texel %d: Dy %g, want %g", i, d[1], want)
		}
	}
}

func vec4At(buf []any, i int) []float64 { return buf[i].([]float64) }

func TestOceanAssembleScalesWithChoppiness(t *testing.T) {
	s := defaultSpec(2, 64)
	s.seed = 6
	h0, omega := runOceanSpectrum(t, s)
	a, b := fieldBuffers(runOceanEvolve(t, s, h0, omega, 5))
	n := s.n
	const lambda = 0.8
	disp, slope := runOceanAssemble(t, n, a, b, lambda, run.Options{})
	var minJ = math.Inf(1)
	for i := 0; i < n*n; i++ {
		wd, ws := assembled(a, b, i, lambda)
		for c := 0; c < 4; c++ {
			if got, want := disp.Data[4*i+c], roundHalf(wd[c]); got != want {
				t.Fatalf("texel %d disp[%d] = %g, want %g", i, c, got, want)
			}
			if got, want := slope.Data[4*i+c], roundHalf(ws[c]); got != want {
				t.Fatalf("texel %d slope[%d] = %g, want %g", i, c, got, want)
			}
		}
		sl, d := slope.Data[4*i:4*i+4], disp.Data[4*i:4*i+4]
		minJ = math.Min(minJ, (1+sl[2])*(1+sl[3])-d[3]*d[3])
	}
	t.Logf("cascade 2 at lambda %.1f: minimum Jacobian %.3f", lambda, minJ)
	if minJ >= 1 {
		t.Errorf("Jacobian never drops below 1 with lambda > 0")
	}
}

// TestOceanAssembleFloat32Half runs the kernel in float32 mode on all four
// default cascades and checks the rgba16float rounding stays within 1e-3
// relative of the ideal values (half precision has 11 significant bits).
func TestOceanAssembleFloat32Half(t *testing.T) {
	const lambda = 0.8
	var worst, biggest float64
	for c := 0; c < 4; c++ {
		s := defaultSpec(c, 64)
		s.seed = 9
		h0, omega := runOceanSpectrum(t, s)
		a, b := fieldBuffers(runOceanEvolve(t, s, h0, omega, 11.5))
		disp, slope := runOceanAssemble(t, s.n, a, b, lambda, run.Options{Float32: true})
		for i := 0; i < s.n*s.n; i++ {
			wd, ws := assembled(a, b, i, lambda)
			for k := 0; k < 4; k++ {
				for _, p := range [][2]float64{{disp.Data[4*i+k], wd[k]}, {slope.Data[4*i+k], ws[k]}} {
					biggest = math.Max(biggest, math.Abs(p[1]))
					err := math.Abs(p[0] - p[1])
					if bound := 1e-3*math.Abs(p[1]) + 1e-7; err > bound {
						t.Fatalf("cascade %d texel %d lane %d: half value %g vs ideal %g", c, i, k, p[0], p[1])
					}
					if r := err / math.Abs(p[1]); math.Abs(p[1]) >= 6.2e-5 && r > worst {
						worst = r
					}
				}
			}
		}
	}
	t.Logf("float32 mode + half rounding: worst relative error %.3g over values in the half normal range (bound 1e-3; smaller values are checked to 1e-7 absolute); largest value %.3g (half max 65504)", worst, biggest)
	if biggest > 65504 {
		t.Errorf("value %g overflows half precision", biggest)
	}
}

// roundHalf rounds v to the nearest IEEE 754 binary16 value (ties to even), the
// precision of an rgba16float texel; it mirrors the interpreter's texture store.
func roundHalf(v float64) float64 {
	if v == 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	a := math.Abs(v)
	var r float64
	if a < math.Ldexp(1, -14) {
		r = math.Ldexp(math.RoundToEven(math.Ldexp(a, 24)), -24)
	} else {
		_, exp := math.Frexp(a)
		shift := 10 - (exp - 1)
		r = math.Ldexp(math.RoundToEven(math.Ldexp(a, shift)), -shift)
	}
	if r > 65504 {
		r = math.Inf(1)
	}
	return math.Copysign(r, v)
}
