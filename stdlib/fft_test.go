package stdlib

import (
	"math"
	"math/cmplx"
	"math/rand"
)

// dft2DInverse is the float64 reference: the direct, unnormalized inverse 2D DFT
//
//	out[z*n+x] = sum over (kz,kx) of in[kz*n+kx] * exp(+2*pi*i*(kx*x + kz*z)/n)
//
// computed as a separable row/column sum (O(n^3)), which is the same transform.
func dft2DInverse(in [][2]complex128, n int) [][2]complex128 {
	tw := make([]complex128, n)
	for k := range tw {
		tw[k] = cmplx.Exp(complex(0, 2*math.Pi*float64(k)/float64(n)))
	}
	tmp := make([][2]complex128, n*n)
	// rows: transform along x
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			var s [2]complex128
			for k := 0; k < n; k++ {
				w := tw[(k*x)%n]
				s[0] += in[z*n+k][0] * w
				s[1] += in[z*n+k][1] * w
			}
			tmp[z*n+x] = s
		}
	}
	out := make([][2]complex128, n*n)
	// columns: transform along z
	for x := 0; x < n; x++ {
		for z := 0; z < n; z++ {
			var s [2]complex128
			for k := 0; k < n; k++ {
				w := tw[(k*z)%n]
				s[0] += tmp[k*n+x][0] * w
				s[1] += tmp[k*n+x][1] * w
			}
			out[z*n+x] = s
		}
	}
	return out
}

// randomSpectrum returns n*n elements of two complex numbers each, with
// components uniform in [-1, 1) from a seeded stream.
func randomSpectrum(n int, seed int64) [][2]complex128 {
	r := rand.New(rand.NewSource(seed))
	v := make([][2]complex128, n*n)
	c := func() complex128 { return complex(r.Float64()*2-1, r.Float64()*2-1) }
	for i := range v {
		v[i] = [2]complex128{c(), c()}
	}
	return v
}

// packVec4 converts complex pairs to the interpreter's array<vec4<f32>> form.
func packVec4(v [][2]complex128) []any {
	out := make([]any, len(v))
	for i, e := range v {
		out[i] = []float64{real(e[0]), imag(e[0]), real(e[1]), imag(e[1])}
	}
	return out
}

func unpackVec4(b []any) [][2]complex128 {
	out := make([][2]complex128, len(b))
	for i, e := range b {
		f := e.([]float64)
		out[i] = [2]complex128{complex(f[0], f[1]), complex(f[2], f[3])}
	}
	return out
}

func rmsOf(v [][2]complex128) float64 {
	var s float64
	for _, e := range v {
		s += real(e[0])*real(e[0]) + imag(e[0])*imag(e[0]) + real(e[1])*real(e[1]) + imag(e[1])*imag(e[1])
	}
	return math.Sqrt(s / float64(4*len(v)))
}

func maxErr(a, b [][2]complex128) float64 {
	var m float64
	for i := range a {
		for s := 0; s < 2; s++ {
			if d := cmplx.Abs(a[i][s] - b[i][s]); d > m {
				m = d
			}
		}
	}
	return m
}
