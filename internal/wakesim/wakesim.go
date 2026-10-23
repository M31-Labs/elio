// Package wakesim is a plain-Go reference of the ship-wake model and the
// Kelvin half-angle metric, for tests. It uses the exact operator |k| on a
// periodic window with a sponge band and a window that slides with the hull,
// so tests can check the metric apart from the Elio kernels, and the wake
// kernels and graph rules against it. It is not used at run time.
package wakesim

import (
	"math"
	"math/cmplx"
)

// The reference scenario in simulation units: a window of N cells, cell = 1 and
// g = 1 (a speed v is in sqrt(g cell) units and the transverse wavelength is
// 2 pi v^2 cells), dt = DT, light damping 0.01, a Sponge-cell absorbing band
// with peak rate 0.8, and Steps steps.
const (
	N      = 128
	Sponge = 12
	DT     = 0.5
	Steps  = 320 // 160 s
)

func sat(x float64) float64 { return math.Min(math.Max(x, 0), 1) }

// HalfAngle measures the half-angle (degrees) of the wake's outer energy
// edge from a window field h (h[z*n+x], x increasing ahead of the hull) with
// the hull at window cell (hullX, n/2). For each distance d behind the hull it
// takes the energy of h^2 summed along x over one transverse wavelength, smooths
// it across the track, finds the outermost local maximum above 15% of the
// cross-section's peak, and the cross-track position beyond it where the energy
// falls to 45% of that lobe's peak. That 45% point marks the cusp: the Airy
// function Ai(0)^2 is 45% of its first maximum, so the edge sits at the
// geometric caustic whatever the smearing at finite distance. The half-angle is
// atan of the least-squares slope (with intercept) of that edge against d over
// d in [dMin, dMax]. margin is the sponge width in cells, excluded at the
// window edges. It returns the angle and the number of cross-sections used.
func HalfAngle(h []float64, n int, hullX, lambdaT, dMin, dMax float64, margin int) (float64, int) {
	half := int(math.Max(3, math.Round(lambdaT/2)))
	var ds, ys []float64
	for d := dMin; d <= dMax; d += 3 {
		xi := int(hullX - d)
		if xi < margin+half {
			break
		}
		e := make([]float64, n)
		for z := n/2 + 1; z < n-margin; z++ {
			for dd := -half; dd <= half; dd++ {
				e[z] += h[z*n+xi+dd] * h[z*n+xi+dd]
			}
		}
		s := make([]float64, n)
		peak := 0.0
		for z := n/2 + 3; z < n-margin-2; z++ {
			s[z] = (e[z-1] + 2*e[z] + e[z+1]) / 4
			peak = math.Max(peak, s[z])
		}
		pz := -1
		for z := n - margin - 3; z > n/2+3; z-- {
			if s[z] >= 0.15*peak && s[z] >= s[z-1] && s[z] >= s[z+1] {
				pz = z
				break
			}
		}
		if pz < 0 {
			continue
		}
		edge := float64(pz)
		for z := pz; z < n-margin-2; z++ {
			if s[z+1] < 0.45*s[pz] {
				edge = float64(z) + (s[z]-0.45*s[pz])/(s[z]-s[z+1])
				break
			}
		}
		ds = append(ds, d)
		ys = append(ys, edge-float64(n/2))
	}
	var sd, sy, sdd, sdy float64
	m := float64(len(ds))
	for i := range ds {
		sd += ds[i]
		sy += ys[i]
		sdd += ds[i] * ds[i]
		sdy += ds[i] * ys[i]
	}
	slope := (m*sdy - sd*sy) / (m*sdd - sd*sd)
	return math.Atan(slope) * 180 / math.Pi, len(ds)
}

// Reference is an independent plain-Go implementation of the same model
// with the exact operator |k|: the window is a plain array that slides.
func Reference(v, srcLength, srcBeam float64) []float64 {
	n := N
	h, hp := make([]float64, n*n), make([]float64, n*n)
	hullX, x0 := float64(n-Sponge-10), float64(n-Sponge-10)
	for step := 0; step < Steps; step++ {
		depth := 0.5 * math.Min(1, float64(step)*DT/30)
		s := make([]float64, n*n)
		in := make([]any, n*n)
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				u, w := (float64(x)+0.5-hullX)/(srcLength/2), (float64(z)+0.5-float64(n)/2)/(srcBeam/2)
				q := u*u + w*w
				head := 0.0
				if q < 1 {
					head = depth * (1 - q) * (1 - q)
				}
				s[z*n+x] = h[z*n+x] + head
				in[z*n+x] = []float64{s[z*n+x], 0, 0, 0}
			}
		}
		q := inverse(n, symbol(n, float64(n), inverse(n, in)))
		next := make([]float64, n*n)
		for z := 0; z < n; z++ {
			for x := 0; x < n; x++ {
				e := min(min(x, z), min(n-1-x, n-1-z))
				tt := sat((Sponge - float64(e)) / Sponge)
				beta := 1 - (0.01+0.8*tt*tt)*DT
				next[z*n+x] = h[z*n+x] + beta*(h[z*n+x]-hp[z*n+x]) - DT*DT*q[z*n+x].([]float64)[0]
			}
		}
		hp, h = h, next
		hullX += v * DT
		for hullX > x0+1 {
			for _, a := range [][]float64{h, hp} {
				for z := 0; z < n; z++ {
					copy(a[z*n:z*n+n-1], a[z*n+1:z*n+n])
					a[z*n+n-1] = 0
				}
			}
			hullX--
		}
	}
	return h
}

// ReferenceAngle runs the exact-operator reference for a hull speed v
// (sqrt(g cell) units) with a pressure patch srcLength x srcBeam cells.
func ReferenceAngle(v, srcLength, srcBeam float64) (float64, int) {
	h := Reference(v, srcLength, srcBeam)
	return HalfAngle(h, N, float64(N-Sponge-10), 2*math.Pi*v*v, 14, 60, Sponge)
}

// inverse is the unnormalized inverse 2D FFT (e^{+i}) of the first complex slot of a packed vec4 buffer.
func inverse(n int, buf []any) []any {
	a := make([]complex128, n*n)
	for i := range buf {
		f := buf[i].([]float64)
		a[i] = complex(f[0], f[1])
	}
	line := make([]complex128, n)
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
	out := make([]any, n*n)
	for i := range out {
		out[i] = []float64{real(a[i]), imag(a[i]), 0, 0}
	}
	return out
}

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

// symbol is wake_symbol: conj(u) |k| / n^2, with k = 0 and the Nyquist lines zeroed.
func symbol(n int, length float64, u []any) []any {
	out := make([]any, n*n)
	for z := 0; z < n; z++ {
		for x := 0; x < n; x++ {
			sx, sz := x, z
			if sx > n/2 {
				sx -= n
			}
			if sz > n/2 {
				sz -= n
			}
			gain := 0.0
			if x != n/2 && z != n/2 {
				gain = 2 * math.Pi / length * math.Hypot(float64(sx), float64(sz)) / float64(n*n)
			}
			c := u[z*n+x].([]float64)
			out[z*n+x] = []float64{gain * c[0], -gain * c[1], 0, 0}
		}
	}
	return out
}
