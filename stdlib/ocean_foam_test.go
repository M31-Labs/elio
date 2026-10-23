package stdlib

import (
	"fmt"
	"math"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// foamParams mirrors OceanFoamParams.
type foamParams struct {
	width, cascades  int
	originX, originZ int // world cell index of the window origin after this step
	shiftX, shiftZ   int // whole texels the origin moved this step
	dt, tau          float64
	threshold, gain  float64
	lambda, cell     float64
	n                [4]int
	tile             [4]float64
	wake             wakeFoamInput
}

// wakeFoamInput is the wake coupling of ocean_foam: the wake window's state
// (turbulence in the z lane of each vec4), its size and cell, and where the foam
// window sits relative to it.
type wakeFoamInput struct {
	width            int
	gain, cell       float64
	relX, relZ       float64 // foam window origin in metres from the wake window's origin corner
	originX, originZ int     // wake window origin cell index, wrapped to [0, width)
	state            [][4]float64
}

func (p foamParams) uniform() map[string]any {
	u := map[string]any{
		"width": int64(p.width), "cascades": int64(p.cascades), "_pad0": int64(0), "_pad1": int64(0),
		"originX": int64(p.originX), "originZ": int64(p.originZ), "shiftX": int64(p.shiftX), "shiftZ": int64(p.shiftZ),
		"dt": p.dt, "tau": p.tau, "threshold": p.threshold, "gain": p.gain,
		"lambda": p.lambda, "cell": p.cell, "_pad2": 0.0, "_pad3": 0.0,
		"n0": int64(p.n[0]), "n1": int64(p.n[1]), "n2": int64(p.n[2]), "n3": int64(p.n[3]),
		"size0": p.tile[0], "size1": p.tile[1], "size2": p.tile[2], "size3": p.tile[3],
		"wakeWidth": int64(p.wake.width), "wakeGain": p.wake.gain, "wakeCell": p.wake.cell, "_pad4": 0.0,
		"wakeRelX": p.wake.relX, "wakeRelZ": p.wake.relZ, "wakeOriginX": int64(p.wake.originX), "wakeOriginZ": int64(p.wake.originZ),
	}
	// The kernel samples cascade c at omX/omZ (the window origin in metres
	// wrapped into the tile, computed here in float64 as the graph does) plus
	// the local texel offset, so it never forms a large coordinate.
	for c := 0; c < 4; c++ {
		u[fmt.Sprintf("omX%d", c)] = wrapTile(float64(p.originX)*p.cell, p.tile[c])
		u[fmt.Sprintf("omZ%d", c)] = wrapTile(float64(p.originZ)*p.cell, p.tile[c])
	}
	return u
}

// wrapTile is a mod size in [0, size), computed in float64; 0 for an unused
// cascade (size 0).
func wrapTile(a, size float64) float64 {
	if size == 0 {
		return 0
	}
	m := math.Mod(a, size)
	if m < 0 {
		m += size
	}
	return m
}

// foamBuffers are the post-FFT B buffers of the cascades: fieldB[c] holds n*n
// vec4 (dDy/dz, dDx/dx, dDz/dz, dDx/dz), unscaled by choppiness.
type foamBuffers [4][][4]float64

// runOceanFoam runs one foam step and returns the new state and the texture.
func runOceanFoam(t *testing.T, p foamParams, state []float64, fb foamBuffers, opts run.Options) ([]float64, *run.Texture) {
	t.Helper()
	m, err := OceanFoam()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	if errs := sema.CheckGather(m, "ocean_foam"); len(errs) != 0 {
		t.Fatalf("CheckGather: %v", errs)
	}
	w := p.width
	out := make([]float64, w*w)
	for i := range out {
		out[i] = 9 // sentinel: every texel must be written
	}
	tex := &run.Texture{Width: w, Height: w, Format: "rgba16float", Data: make([]float64, w*w*4)}
	vars := map[string]any{"foam": p.uniform(), "foamIn": append([]float64(nil), state...), "foamOut": out, "foamTex": tex}
	if p.wake.width > 0 {
		vars["wakeIn"] = vec4Buf(p.wake.state)
	} else {
		vars["wakeIn"] = []any{[]float64{0, 0, 0, 0}}
	}
	for c := 0; c < 4; c++ {
		n := p.n[c]
		if n == 0 {
			n = 1
		}
		buf := make([]any, n*n)
		for i := range buf {
			v := [4]float64{}
			if i < len(fb[c]) {
				v = fb[c][i]
			}
			buf[i] = []float64{v[0], v[1], v[2], v[3]}
		}
		vars["fieldB"+string(rune('0'+c))] = buf
	}
	if err := run.RunWith(m, "ocean_foam", w*w, &run.Memory{Vars: vars}, opts); err != nil {
		t.Fatal(err)
	}
	return out, tex
}

func posMod(a, n int) int { return ((a % n) + n) % n }

func constState(w int, v float64) []float64 {
	s := make([]float64, w*w)
	for i := range s {
		s[i] = v
	}
	return s
}

func TestOceanFoamDecays(t *testing.T) {
	p := foamParams{width: 16, dt: 1.0 / 60, tau: 2, threshold: 0.5, gain: 0, lambda: 0.8, cell: 0.5}
	state := constState(16, 1)
	var tex *run.Texture
	for step := 0; step < 60; step++ {
		state, tex = runOceanFoam(t, p, state, foamBuffers{}, run.Options{})
	}
	want := math.Exp(-60 * p.dt / p.tau)
	for i, v := range state {
		if math.Abs(v-want) > 0.01*want {
			t.Fatalf("texel %d = %g after 60 steps, want %g within 1%%", i, v, want)
		}
	}
	if got := tex.Data[0]; math.Abs(got-want) > 0.01*want {
		t.Errorf("texture foam %g, want %g", got, want)
	}
	t.Logf("60 steps of dt = 1/60 s, tau = 2 s: foam %.6f, exp(-0.5) = %.6f", state[0], want)
}

func TestOceanFoamToroidalShift(t *testing.T) {
	const w = 16
	for _, sh := range []struct{ dx, dz int }{{3, -2}, {-5, 4}, {0, 0}, {w, 1}} {
		p := foamParams{width: w, dt: 0, tau: 2, threshold: 0.5, gain: 0, lambda: 0.8, cell: 0.5, originX: 20, originZ: -5, shiftX: sh.dx, shiftZ: sh.dz}
		state := constState(w, 1)
		// Mark one world cell that stays in the window after the shift.
		oldX, oldZ := p.originX-sh.dx, p.originZ-sh.dz
		markU, markV := posMod(oldX+8, w), posMod(oldZ+8, w)
		state[markV*w+markU] = 0.7
		got, _ := runOceanFoam(t, p, state, foamBuffers{}, run.Options{})
		cleared := 0
		for v := 0; v < w; v++ {
			for u := 0; u < w; u++ {
				ru, rv := posMod(u-p.originX, w), posMod(v-p.originZ, w)
				enteringX := (sh.dx > 0 && ru >= w-sh.dx) || (sh.dx < 0 && ru < -sh.dx)
				enteringZ := (sh.dz > 0 && rv >= w-sh.dz) || (sh.dz < 0 && rv < -sh.dz)
				want := state[v*w+u]
				if enteringX || enteringZ {
					want = 0
					cleared++
				}
				if got[v*w+u] != want {
					t.Fatalf("shift (%d,%d): texel (%d,%d) = %g, want %g", sh.dx, sh.dz, u, v, got[v*w+u], want)
				}
			}
		}
		// The marked world cell keeps its buffer slot, so its position relative to
		// the window origin moves by (-dx, -dz).
		relBefore := [2]int{posMod(markU-oldX, w), posMod(markV-oldZ, w)}
		relAfter := [2]int{posMod(markU-p.originX, w), posMod(markV-p.originZ, w)}
		if relAfter != [2]int{posMod(relBefore[0]-sh.dx, w), posMod(relBefore[1]-sh.dz, w)} {
			t.Errorf("shift (%d,%d): marked cell window position %v -> %v", sh.dx, sh.dz, relBefore, relAfter)
		}
		t.Logf("shift (%d,%d): %d of %d texels cleared", sh.dx, sh.dz, cleared, w*w)
	}
}

// bilinear4 samples an n x n periodic tile of vec4 at world (px, pz).
func bilinear4(d [][4]float64, n int, tile, px, pz float64) [4]float64 {
	u, v := px/tile, pz/tile
	fx, fz := (u-math.Floor(u))*float64(n), (v-math.Floor(v))*float64(n)
	x0, z0 := min(int(math.Floor(fx)), n-1), min(int(math.Floor(fz)), n-1)
	rx, rz := fx-math.Floor(fx), fz-math.Floor(fz)
	x1, z1 := (x0+1)%n, (z0+1)%n
	var o [4]float64
	for k := 0; k < 4; k++ {
		o[k] = (1-rz)*((1-rx)*d[z0*n+x0][k]+rx*d[z0*n+x1][k]) + rz*((1-rx)*d[z1*n+x0][k]+rx*d[z1*n+x1][k])
	}
	return o
}

func TestOceanFoamInjectsFromCombinedJacobian(t *testing.T) {
	const w = 16
	// Two cascades with different sizes and tiles, varying fields, so the world
	// mapping, the cascade sums and the saturation are all exercised.
	var fb foamBuffers
	n0, n1 := 8, 4
	fb[0] = make([][4]float64, n0*n0)
	fb[1] = make([][4]float64, n1*n1)
	for z := 0; z < n0; z++ {
		for x := 0; x < n0; x++ {
			fb[0][z*n0+x] = [4]float64{0, 0.5 * math.Sin(2*math.Pi*float64(x)/8), -0.4 * math.Cos(2*math.Pi*float64(z)/8), 0.2 * math.Sin(2*math.Pi*float64(x+z)/8)}
		}
	}
	for z := 0; z < n1; z++ {
		for x := 0; x < n1; x++ {
			fb[1][z*n1+x] = [4]float64{0, 0.3 * math.Cos(2*math.Pi*float64(x)/4), 0.25 * math.Sin(2*math.Pi*float64(z)/4), 0.1}
		}
	}
	p := foamParams{width: w, cascades: 2, originX: -3, originZ: 11, dt: 0.1, tau: 4, threshold: 0.9, gain: 3, lambda: 1.2, cell: 0.5,
		n: [4]int{n0, n1}, tile: [4]float64{4, 3}}
	jac := func(wx, wz int) float64 {
		px, pz := (float64(wx)+0.5)*p.cell, (float64(wz)+0.5)*p.cell
		var sxx, szz, sxz float64
		for c := 0; c < 2; c++ {
			b := bilinear4(fb[c], p.n[c], p.tile[c], px, pz)
			sxx, szz, sxz = sxx+b[1], szz+b[2], sxz+b[3]
		}
		return (1+p.lambda*sxx)*(1+p.lambda*szz) - (p.lambda*sxz)*(p.lambda*sxz)
	}
	state := constState(w, 0.4)
	got, tex := runOceanFoam(t, p, state, fb, run.Options{})
	var lo, hi, saturated = 1.0, 0.0, 0
	decay := math.Exp(-p.dt / p.tau)
	for v := 0; v < w; v++ {
		for u := 0; u < w; u++ {
			wx, wz := p.originX+posMod(u-p.originX, w), p.originZ+posMod(v-p.originZ, w)
			j := jac(wx, wz)
			want := math.Min(0.4*decay+math.Min(math.Max((p.threshold-j)*p.gain, 0), 1), 1)
			if math.Abs(got[v*w+u]-want) > 1e-12 {
				t.Fatalf("texel (%d,%d) world (%d,%d): foam %.12g, want %.12g (J %.6g)", u, v, wx, wz, got[v*w+u], want, j)
			}
			if tx := tex.Data[4*(v*w+u):]; tx[0] != roundHalf(want) || tx[1] != roundHalf(j) || tx[2] != 0 || tx[3] != 1 {
				t.Fatalf("texel (%d,%d): texture %v, want (%g, %g, 0, 1)", u, v, tx[:4], roundHalf(want), roundHalf(j))
			}
			lo, hi = math.Min(lo, want), math.Max(hi, want)
			if want == 1 {
				saturated++
			}
		}
	}
	t.Logf("foam after one step ranges %.3f to %.3f; %d of %d texels saturated", lo, hi, saturated, w*w)
	if hi-lo < 0.1 {
		t.Errorf("injection barely varies (%g to %g): the test does not exercise the Jacobian", lo, hi)
	}
}

// wakeFoamNoise mirrors the kernel's streak noise at world metres (gx, gz) for streak angle theta.
func wakeFoamNoise(gx, gz, theta float64) float64 {
	fwx, fwz := math.Sin(theta), math.Cos(theta)
	ga, gb := gx*fwx+gz*fwz, gx*fwz-gz*fwx
	hash := func(a, b, ph float64) float64 {
		v := math.Sin(12.9898*a+78.233*b+ph) * 43758.5453
		return v - math.Floor(v)
	}
	sum := 0.0
	for o, oc := range [][3]float64{{16, 1.6, 0.35}, {6, 3.5, 0.4}, {1.4, 1.4, 0.25}} {
		qa, qb := ga/oc[0], gb/oc[1]
		ia, ib := math.Floor(qa), math.Floor(qb)
		ta, tb := qa-ia, qb-ib
		ea, eb := ta*ta*(3-2*ta), tb*tb*(3-2*tb)
		ma, mb := ia-289*math.Floor(ia/289), ib-289*math.Floor(ib/289)
		ph := 37.719 * float64(o)
		mix := func(a, b, t float64) float64 { return a*(1-t) + b*t }
		sum += oc[2] * mix(mix(hash(ma, mb, ph), hash(ma+1, mb, ph), ea), mix(hash(ma, mb+1, ph), hash(ma+1, mb+1, ph), ea), eb)
	}
	return sum
}

func smoothstep01(lo, hi, x float64) float64 {
	t := math.Min(math.Max((x-lo)/(hi-lo), 0), 1)
	return t * t * (3 - 2*t)
}

// With a wake, the texture's wake lane is the turbulence T (bilinear at the foam
// cell centre, 0 outside the wake window) compared with a streaky noise
// threshold; the foam state itself holds crest foam only. With gain 0 there is
// no wake foam.
func TestOceanFoamFollowsWakeTurbulence(t *testing.T) {
	const w, ww = 16, 8
	wake := wakeFoamInput{width: ww, gain: 1.5, cell: 2, originX: 5, originZ: 3, relX: -1.3, relZ: 0.9, state: make([][4]float64, ww*ww)}
	// Turbulence by window cell (cu, cv): a ridge along cu = 3, stored toroidally.
	turb := func(cu, cv int) float64 {
		if cu < 0 || cu >= ww || cv < 0 || cv >= ww {
			return 0
		}
		return 0.7 * math.Exp(-float64((cu-3)*(cu-3))/2) * (0.4 + 0.6*float64(cv)/ww)
	}
	const streakAngle = 1.1
	for cv := 0; cv < ww; cv++ {
		for cu := 0; cu < ww; cu++ {
			wake.state[posMod(cv+wake.originZ, ww)*ww+posMod(cu+wake.originX, ww)] = [4]float64{9, 9, turb(cu, cv), streakAngle}
		}
	}
	p := foamParams{width: w, dt: 0.1, tau: 4, threshold: 0.9, gain: 0, lambda: 0.8, cell: 0.5, originX: 7, originZ: -2, wake: wake}
	state := constState(w, 0.05)
	got, tex := runOceanFoam(t, p, state, foamBuffers{}, run.Options{})
	decay := math.Exp(-p.dt / p.tau)
	var covered, partial int
	for v := 0; v < w; v++ {
		for u := 0; u < w; u++ {
			ru, rv := posMod(u-p.originX, w), posMod(v-p.originZ, w)
			px := (wake.relX+(float64(ru)+0.5)*p.cell)/wake.cell - 0.5
			pz := (wake.relZ+(float64(rv)+0.5)*p.cell)/wake.cell - 0.5
			x0, z0 := math.Floor(px), math.Floor(pz)
			fx, fz := px-x0, pz-z0
			T := (1-fz)*((1-fx)*turb(int(x0), int(z0))+fx*turb(int(x0)+1, int(z0))) + fz*((1-fx)*turb(int(x0), int(z0)+1)+fx*turb(int(x0)+1, int(z0)+1))
			theta := 0.0
			if T > 0 {
				theta = streakAngle // some corner inside the window with weight >= 0.25 carries the angle
			}
			gx, gz := (float64(p.originX+ru)+0.5)*p.cell, (float64(p.originZ+rv)+0.5)*p.cell
			nn := math.Min(math.Max((wakeFoamNoise(gx, gz, theta)-0.25)/0.5, 0), 1)
			wf := smoothstep01(0, 0.4, wake.gain*T-(0.12+0.76*nn))
			want := 0.05 * decay
			if math.Abs(got[v*w+u]-want) > 1e-12 {
				t.Fatalf("texel (%d,%d): foam state %.12g, want crest foam only, %.12g", u, v, got[v*w+u], want)
			}
			if tx := tex.Data[4*(v*w+u):]; math.Abs(tx[2]-wf) > 2e-3 || tx[0] != roundHalf(want) {
				t.Fatalf("texel (%d,%d): texture (%g, _, %g), want (%g, _, %g) (turbulence %.4g)", u, v, tx[0], tx[2], roundHalf(want), wf, T)
			}
			if wf > 0.5 {
				covered++
			} else if wf > 0.01 {
				partial++
			}
		}
	}
	t.Logf("%d of %d foam texels fully covered by wake foam, %d partly", covered, w*w, partial)
	if covered == 0 || covered == w*w || partial == 0 {
		t.Errorf("wake foam covers %d texels and edges %d: the test does not exercise the threshold", covered, partial)
	}
	// gain 0: no wake foam.
	p.wake.gain = 0
	plain, ptex := runOceanFoam(t, p, state, foamBuffers{}, run.Options{})
	for i, f := range plain {
		if f != 0.05*decay || ptex.Data[4*i+2] != 0 {
			t.Fatalf("gain 0: texel %d = %g (wake lane %g), want %g and 0", i, f, ptex.Data[4*i+2], 0.05*decay)
		}
	}
}
