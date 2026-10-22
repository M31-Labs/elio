package stdlib

import (
	"math/rand"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func runOceanMip(t *testing.T, sw, sh int, src [][4]float64) *run.Texture {
	t.Helper()
	m, err := OceanMip()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.Check(m); len(errs) != 0 {
		t.Fatalf("sema: %v", sema.Errors(errs))
	}
	if errs := sema.CheckGather(m, "ocean_mip"); len(errs) != 0 {
		t.Fatalf("CheckGather: %v", errs)
	}
	dw, dh := (sw+1)/2, (sh+1)/2
	buf := make([]any, len(src))
	for i, v := range src {
		buf[i] = []float64{v[0], v[1], v[2], v[3]}
	}
	tex := &run.Texture{Width: dw, Height: dh, Format: "rgba16float", Data: make([]float64, dw*dh*4)}
	mem := &run.Memory{Vars: map[string]any{
		"mip": map[string]any{"srcWidth": int64(sw), "srcHeight": int64(sh), "dstWidth": int64(dw), "dstHeight": int64(dh)},
		"src": buf, "dst": tex,
	}}
	if err := run.Run(m, "ocean_mip", dw*dh, mem); err != nil {
		t.Fatal(err)
	}
	return tex
}

func TestOceanMipConstantStaysConstant(t *testing.T) {
	const w = 16
	c := [4]float64{0.3, -1.25, 17.5, 0.0001}
	src := make([][4]float64, w*w)
	for i := range src {
		src[i] = c
	}
	tex := runOceanMip(t, w, w, src)
	for i := 0; i < (w/2)*(w/2); i++ {
		for k := 0; k < 4; k++ {
			if got, want := tex.Data[4*i+k], roundHalf(c[k]); got != want {
				t.Fatalf("texel %d lane %d = %g, want %g", i, k, got, want)
			}
		}
	}
}

func TestOceanMipCheckerboardAveragesToMean(t *testing.T) {
	const w = 16
	src := make([][4]float64, w*w)
	for y := 0; y < w; y++ {
		for x := 0; x < w; x++ {
			v := float64((x + y) % 2) // 0 and 1, so the mean is 0.5
			src[y*w+x] = [4]float64{v, 1 - v, 2 * v, 1}
		}
	}
	tex := runOceanMip(t, w, w, src)
	for i := 0; i < (w/2)*(w/2); i++ {
		want := [4]float64{0.5, 0.5, 1, 1}
		for k := 0; k < 4; k++ {
			if got := tex.Data[4*i+k]; got != want[k] {
				t.Fatalf("texel %d lane %d = %g, want %g", i, k, got, want[k])
			}
		}
	}
}

// TestOceanMipMatchesBoxFilter checks random data on a non-square level, including
// the clamped odd edge.
func TestOceanMipMatchesBoxFilter(t *testing.T) {
	const sw, sh = 9, 6
	r := rand.New(rand.NewSource(5))
	src := make([][4]float64, sw*sh)
	for i := range src {
		for k := 0; k < 4; k++ {
			src[i][k] = r.Float64()*8 - 4
		}
	}
	tex := runOceanMip(t, sw, sh, src)
	dw, dh := (sw+1)/2, (sh+1)/2
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			x1, y1 := min(2*x+1, sw-1), min(2*y+1, sh-1)
			for k := 0; k < 4; k++ {
				want := (src[2*y*sw+2*x][k] + src[2*y*sw+x1][k] + src[y1*sw+2*x][k] + src[y1*sw+x1][k]) / 4
				if got := tex.Data[4*(y*dw+x)+k]; got != roundHalf(want) {
					t.Fatalf("(%d,%d) lane %d = %g, want %g (half of %g)", x, y, k, got, roundHalf(want), want)
				}
			}
		}
	}
}
