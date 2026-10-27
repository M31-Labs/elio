package run

import (
	"fmt"
	"math"
)

// Texture is a 2D texture for the interpreter: a write-only storage texture
// (ir.StorageTexture) or a sampled one (ir.Texture2D). Put a *Texture in
// Memory.Vars under the binding's name.
type Texture struct {
	Width, Height int
	Format        string    // as ir.StorageTexture.Format
	Data          []float64 // Width*Height*4, row-major, RGBA; r32float uses channel 0

	// Levels holds the mip levels below the texture itself, for a sampled
	// texture: level 0 is this texture, level L > 0 is Levels[L-1]. The chain is
	// optional and the caller fills it; the interpreter does not build or resize
	// it, and each level carries its own Width, Height and Data. textureLoad and
	// textureDimensions with a level the chain lacks return zeros.
	Levels []*Texture
}

// level returns mip level l (0 is t itself), or nil if the chain lacks it.
func (t *Texture) level(l int) *Texture {
	if l == 0 {
		return t
	}
	if l < 0 || l > len(t.Levels) {
		return nil
	}
	return t.Levels[l-1]
}

// load reads texel (x, y), or zeros when it is outside the texture, as WebGPU
// textureLoad does.
func (t *Texture) load(x, y int) []float64 {
	out := make([]float64, 4)
	if x < 0 || y < 0 || x >= t.Width || y >= t.Height || len(t.Data) < t.Width*t.Height*4 {
		return out
	}
	i := (y*t.Width + x) * 4
	copy(out, t.Data[i:i+4])
	if t.Format == "r32float" {
		out[1], out[2], out[3] = 0, 0, 1
	}
	return out
}

// store writes v (up to four channels) at texel (x, y), rounding to the
// format's precision. A write outside the texture is dropped, as in WebGPU.
func (t *Texture) store(x, y int, v []float64) error {
	if len(v) != 4 {
		return fmt.Errorf("run: textureStore needs a vec4 value, got %d components", len(v))
	}
	if len(t.Data) < t.Width*t.Height*4 {
		return fmt.Errorf("run: texture data holds %d values, need %d", len(t.Data), t.Width*t.Height*4)
	}
	if x < 0 || y < 0 || x >= t.Width || y >= t.Height {
		return nil
	}
	i := (y*t.Width + x) * 4
	switch t.Format {
	case "rgba32float", "r32float":
		for c := 0; c < 4; c++ {
			t.Data[i+c] = float64(float32(v[c]))
		}
	case "rgba16float":
		for c := 0; c < 4; c++ {
			t.Data[i+c] = roundHalf(v[c])
		}
	default:
		return fmt.Errorf("run: unknown texture format %q", t.Format)
	}
	return nil
}

// roundHalf rounds v to the nearest IEEE 754 binary16 value (ties to even) and
// returns it as a float64. Values past the half range become +-Inf.
func roundHalf(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) || v == 0 {
		return v
	}
	a := math.Abs(v)
	var r float64
	if a < math.Ldexp(1, -14) {
		// Subnormal range: a fixed step of 2^-24.
		r = math.Ldexp(math.RoundToEven(math.Ldexp(a, 24)), -24)
	} else {
		_, exp := math.Frexp(a) // a = m * 2^exp, m in [0.5, 1)
		shift := 10 - (exp - 1) // step is 2^(e-10) with e = exp-1
		r = math.Ldexp(math.RoundToEven(math.Ldexp(a, shift)), -shift)
	}
	if r > 65504 {
		r = math.Inf(1)
	}
	return math.Copysign(r, v)
}
