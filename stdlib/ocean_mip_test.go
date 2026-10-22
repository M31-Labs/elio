package stdlib

import (
	"math/rand"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// mipSource builds a rgba16float texture from texel values, row-major.
func mipSource(w, h int, v func(x, y int) [4]float64) *run.Texture {
	t := &run.Texture{Width: w, Height: h, Format: "rgba16float", Data: make([]float64, w*h*4)}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for k, c := range v(x, y) {
				t.Data[4*(y*w+x)+k] = roundHalf(c)
			}
		}
	}
	return t
}

// runOceanMip downsamples level srcLevel of src (level 0 is src itself, the
// rest are src.Levels) into a new texture half the size, rounded up.
func runOceanMip(t *testing.T, src *run.Texture, srcLevel int) *run.Texture {
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
	from := src
	if srcLevel > 0 {
		from = src.Levels[srcLevel-1]
	}
	dw, dh := (from.Width+1)/2, (from.Height+1)/2
	dst := &run.Texture{Width: dw, Height: dh, Format: "rgba16float", Data: make([]float64, dw*dh*4)}
	mem := &run.Memory{Vars: map[string]any{
		"mip": map[string]any{"srcLevel": int64(srcLevel), "dstWidth": int64(dw), "dstHeight": int64(dh), "pad": int64(0)},
		"src": src, "dst": dst,
	}}
	if err := run.Run(m, "ocean_mip", dw*dh, mem); err != nil {
		t.Fatal(err)
	}
	return dst
}

func TestOceanMipConstantStaysConstant(t *testing.T) {
	const w = 16
	c := [4]float64{0.3, -1.25, 17.5, 0.0001}
	src := mipSource(w, w, func(x, y int) [4]float64 { return c })
	tex := runOceanMip(t, src, 0)
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
	src := mipSource(w, w, func(x, y int) [4]float64 {
		v := float64((x + y) % 2) // 0 and 1, so the mean is 0.5
		return [4]float64{v, 1 - v, 2 * v, 1}
	})
	tex := runOceanMip(t, src, 0)
	for i := 0; i < (w/2)*(w/2); i++ {
		want := [4]float64{0.5, 0.5, 1, 1}
		for k := 0; k < 4; k++ {
			if got := tex.Data[4*i+k]; got != want[k] {
				t.Fatalf("texel %d lane %d = %g, want %g", i, k, got, want[k])
			}
		}
	}
}

// boxFilter is the reference: the mean of the four texels (2x, 2y) to
// (2x+1, 2y+1), with the second tap clamped to the last column or row.
func boxFilter(src *run.Texture, x, y, k int) float64 {
	w, h := src.Width, src.Height
	x1, y1 := min(2*x+1, w-1), min(2*y+1, h-1)
	at := func(px, py int) float64 { return src.Data[4*(py*w+px)+k] }
	return (at(2*x, 2*y) + at(x1, 2*y) + at(2*x, y1) + at(x1, y1)) / 4
}

// TestOceanMipOddEdgeClamp checks random data on a non-square level with an odd
// width and height, so the last column and row clamp.
func TestOceanMipOddEdgeClamp(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	src := mipSource(9, 7, func(x, y int) [4]float64 {
		return [4]float64{r.Float64()*8 - 4, r.Float64()*8 - 4, r.Float64()*8 - 4, r.Float64()*8 - 4}
	})
	tex := runOceanMip(t, src, 0)
	if tex.Width != 5 || tex.Height != 4 {
		t.Fatalf("level size %dx%d, want 5x4", tex.Width, tex.Height)
	}
	for y := 0; y < tex.Height; y++ {
		for x := 0; x < tex.Width; x++ {
			for k := 0; k < 4; k++ {
				want := boxFilter(src, x, y, k)
				if got := tex.Data[4*(y*tex.Width+x)+k]; got != roundHalf(want) {
					t.Fatalf("(%d,%d) lane %d = %g, want %g (half of %g)", x, y, k, got, roundHalf(want), want)
				}
			}
		}
	}
	// The corner texel (4, 3) has both taps clamped on both axes: it is the
	// source texel (8, 6) itself.
	for k := 0; k < 4; k++ {
		if got, want := tex.Data[4*(3*5+4)+k], src.Data[4*(6*9+8)+k]; got != want {
			t.Errorf("corner lane %d = %g, want source texel %g", k, got, want)
		}
	}
}

// TestOceanMipChain builds a four-level chain by reading level L-1 of one
// texture and filling level L, as the host does, and checks every level against
// the box filter of the level above.
func TestOceanMipChain(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	base := mipSource(13, 10, func(x, y int) [4]float64 {
		return [4]float64{r.Float64() - 0.5, r.Float64() * 100, float64(x), float64(y)}
	})
	levels := []*run.Texture{base}
	for l := 1; l <= 3; l++ {
		next := runOceanMip(t, base, l-1)
		base.Levels = append(base.Levels, next)
		levels = append(levels, next)
	}
	wantSizes := [][2]int{{13, 10}, {7, 5}, {4, 3}, {2, 2}}
	for l, lv := range levels {
		if lv.Width != wantSizes[l][0] || lv.Height != wantSizes[l][1] {
			t.Fatalf("level %d is %dx%d, want %dx%d", l, lv.Width, lv.Height, wantSizes[l][0], wantSizes[l][1])
		}
		if l == 0 {
			continue
		}
		up := levels[l-1]
		for y := 0; y < lv.Height; y++ {
			for x := 0; x < lv.Width; x++ {
				for k := 0; k < 4; k++ {
					if got, want := lv.Data[4*(y*lv.Width+x)+k], roundHalf(boxFilter(up, x, y, k)); got != want {
						t.Fatalf("level %d (%d,%d) lane %d = %g, want %g", l, x, y, k, got, want)
					}
				}
			}
		}
	}
}
