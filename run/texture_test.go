package run

import (
	"math"
	"testing"

	"m31labs.dev/elio/ir"
)

func newTex(w, h int, format string) *Texture {
	return &Texture{Width: w, Height: h, Format: format, Data: make([]float64, w*h*4)}
}

func (t *Texture) texel(x, y int) []float64 {
	i := (y*t.Width + x) * 4
	return t.Data[i : i+4]
}

func TestTextureFill(t *testing.T) {
	tex := newTex(4, 4, "rgba32float")
	mem := &Memory{Vars: map[string]any{"disp": tex}}
	if err := Run(ir.TextureFill("rgba32float"), "main", 16, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := tex.texel(x, y)
			want := []float64{float64(x), float64(y), 0.5, 1}
			for c := range want {
				if got[c] != want[c] {
					t.Errorf("texel(%d,%d) = %v, want %v", x, y, got, want)
					break
				}
			}
		}
	}
}

func TestTextureOutOfRangeWriteDropped(t *testing.T) {
	tex := newTex(4, 4, "rgba32float")
	mem := &Memory{Vars: map[string]any{"disp": tex}}
	// 32 invocations: gid 16..31 map to y = 4..7, outside the 4x4 texture.
	if err := Run(ir.TextureFill("rgba32float"), "main", 32, mem); err != nil {
		t.Fatalf("out-of-range write must not fail the run: %v", err)
	}
	if len(tex.Data) != 64 {
		t.Fatalf("texture grew to %d values", len(tex.Data))
	}
	if got := tex.texel(3, 3); got[0] != 3 || got[1] != 3 {
		t.Errorf("in-range texel disturbed: %v", got)
	}
}

func TestTextureHalfRounding(t *testing.T) {
	m := &ir.Module{
		Bindings: []ir.Binding{{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Write, Name: "t", Type: ir.StorageTexture{Format: "rgba16float"}}},
		Kernels: []ir.Kernel{{Name: "main", WorkgroupSize: [3]int{1, 1, 1}, Body: []ir.Stmt{
			ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{
				ir.Name{Name: "t"}, ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Lit{Text: "0u"}, ir.Lit{Text: "0u"}}},
				ir.Call{Func: "vec4", Args: []ir.Expr{ir.Lit{Text: "0.1"}, ir.Lit{Text: "65519.0"}, ir.Lit{Text: "0.00000005"}, ir.Lit{Text: "1.0"}}},
			}}},
		}}},
	}
	tex := newTex(1, 1, "rgba16float")
	if err := Run(m, "main", 1, &Memory{Vars: map[string]any{"t": tex}}); err != nil {
		t.Fatal(err)
	}
	if got := tex.Data[0]; got != 0.0999755859375 {
		t.Errorf("0.1 as half = %v, want 0.0999755859375", got)
	}
	if got := tex.Data[1]; got != 65504 {
		t.Errorf("65519 as half = %v, want 65504", got)
	}
	if got := tex.Data[2]; got != math.Ldexp(1, -24) {
		t.Errorf("5e-8 as half = %v, want 2^-24", got)
	}
	if tex.Data[3] != 1 {
		t.Errorf("1.0 as half = %v", tex.Data[3])
	}
}

func TestRoundHalf(t *testing.T) {
	for _, c := range []struct{ in, want float64 }{
		{0, 0}, {1, 1}, {-2.5, -2.5}, {65520, math.Inf(1)}, {1e9, math.Inf(1)},
		{2049, 2048},                 // tie rounds to even
		{2051, 2052},                 // tie rounds to even
		{1.0009765625, 1.0009765625}, // 1 + 2^-10 exact
		{math.Ldexp(1, -25), 0},      // half of the smallest subnormal ties to even (0)
	} {
		if got := roundHalf(c.in); got != c.want {
			t.Errorf("roundHalf(%v) = %v, want %v", c.in, got, c.want)
		}
	}
	if !math.IsNaN(roundHalf(math.NaN())) {
		t.Error("NaN must stay NaN")
	}
}

func TestTextureR32AndDimensions(t *testing.T) {
	m := &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Write, Name: "t", Type: ir.StorageTexture{Format: "r32float"}},
		},
		Kernels: []ir.Kernel{{Name: "main", WorkgroupSize: [3]int{1, 1, 1}, Body: []ir.Stmt{
			ir.Let{Name: "d", Value: ir.Call{Func: "textureDimensions", Args: []ir.Expr{ir.Name{Name: "t"}}}},
			ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{
				ir.Name{Name: "t"}, ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Lit{Text: "1u"}, ir.Lit{Text: "0u"}}},
				ir.Call{Func: "vec4", Args: []ir.Expr{ir.Call{Func: "f32", Args: []ir.Expr{ir.Member{E: ir.Name{Name: "d"}, Field: "y"}}}, ir.Lit{Text: "0.0"}, ir.Lit{Text: "0.0"}, ir.Lit{Text: "0.0"}}},
			}}},
		}}},
	}
	tex := newTex(3, 2, "r32float")
	if err := Run(m, "main", 1, &Memory{Vars: map[string]any{"t": tex}}); err != nil {
		t.Fatal(err)
	}
	if got := tex.texel(1, 0)[0]; got != 2 {
		t.Errorf("height via textureDimensions = %v, want 2", got)
	}
}

// mipTex builds a 4x4 rgba32float texture whose texel (x, y) is (x, y, 0, 1)
// with a one-level mip chain: a 2x2 level whose texels hold (10+x, 10+y, 0, 1).
func mipTex() *Texture {
	base := newTex(4, 4, "rgba32float")
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			copy(base.texel(x, y), []float64{float64(x), float64(y), 0, 1})
		}
	}
	l1 := newTex(2, 2, "rgba32float")
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			copy(l1.texel(x, y), []float64{float64(10 + x), float64(10 + y), 0, 1})
		}
	}
	base.Levels = []*Texture{l1}
	return base
}

func TestTextureLoadAndDimensionsLevels(t *testing.T) {
	src := mipTex()
	dst := newTex(4, 4, "rgba32float")
	mem := &Memory{Vars: map[string]any{"src": src, "dst": dst}}
	if err := Run(ir.TextureCopy(), "main", 16, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	// dst = level-0 texel + width of level 1 (2) in x.
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := dst.texel(x, y)
			want := []float64{float64(x) + 2, float64(y), 0, 1}
			for c := range want {
				if got[c] != want[c] {
					t.Fatalf("dst(%d,%d) = %v, want %v", x, y, got, want)
				}
			}
		}
	}
}

func loadModule(level, x, y string) *ir.Module {
	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Texture2D{}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Write, Name: "dst", Type: ir.StorageTexture{Format: "rgba32float"}},
		},
		Kernels: []ir.Kernel{{Name: "main", WorkgroupSize: [3]int{1, 1, 1}, Body: []ir.Stmt{
			ir.Do{Expr: ir.Call{Func: "textureStore", Args: []ir.Expr{
				ir.Name{Name: "dst"}, ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Lit{Text: "0u"}, ir.Lit{Text: "0u"}}},
				ir.Call{Func: "textureLoad", Args: []ir.Expr{ir.Name{Name: "src"},
					ir.Call{Func: "vec2u", Args: []ir.Expr{ir.Lit{Text: x}, ir.Lit{Text: y}}}, ir.Lit{Text: level}}},
			}}},
		}}},
	}
}

func TestTextureLoadLevelsAndBounds(t *testing.T) {
	cases := []struct {
		name        string
		level, x, y string
		want        []float64
	}{
		{"level0", "0u", "3u", "2u", []float64{3, 2, 0, 1}},
		{"level1", "1u", "1u", "0u", []float64{11, 10, 0, 1}},
		{"x out of range", "0u", "4u", "0u", []float64{0, 0, 0, 0}},
		{"level1 out of range", "1u", "2u", "0u", []float64{0, 0, 0, 0}},
		{"missing level", "2u", "0u", "0u", []float64{0, 0, 0, 0}},
	}
	for _, tc := range cases {
		dst := newTex(1, 1, "rgba32float")
		mem := &Memory{Vars: map[string]any{"src": mipTex(), "dst": dst}}
		if err := Run(loadModule(tc.level, tc.x, tc.y), "main", 1, mem); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for c := range tc.want {
			if dst.Data[c] != tc.want[c] {
				t.Errorf("%s: got %v, want %v", tc.name, dst.Data[:4], tc.want)
				break
			}
		}
	}
}
