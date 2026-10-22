package stdlib

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// The swell-wave golden ties the Elio ocean_spectrum kernel to GoSX's
// deterministic Go function scene.OceanSwellWaves. GoSX cannot import Elio
// (the parser would enter its wasm builds), so this test runs the kernel on the
// interpreter, selects the K strongest bins of cascades 0 and 1, and writes
// them to ocean/testdata/swell-bins.json. GoSX keeps a byte-identical copy in
// scene/testdata/ocean-swell-bins.json and checks its Go function against it.
// UPDATE_GOLDEN=1 regenerates the file here; copy it to GoSX afterwards.
//
// Each selected bin becomes a plane wave. With the inverse FFT convention
// e^{+ik.x} and h(k,t) = h0(k) e^{-iwt} + conj h0(-k) e^{+iwt}, the real height
// field is
//
//	sum over all bins of 2 Re[ h0(k) e^{i(k.x - w t)} ]
//
// so bin k is the wave y = A cos(k.x - w t + phase) travelling toward +k with
// A = 2 |h0(k)| and phase = arg h0(k), the (x, y) lane pair.
// TestOceanSwellWavesSumToField checks that identity against a direct sum of the
// kernel's own h(k, t).

const swellGoldenPath = "ocean/testdata/swell-bins.json"

// swellPreset repeats the GoSX scene.OceanPreset values. windDirection is in
// degrees as scene.Ocean.WindDirection: 0 travels toward +Z, positive toward +X.
type swellPreset struct {
	Name           string  `json:"name"`
	WindSpeed      float64 `json:"windSpeed"`
	Fetch          float64 `json:"fetch"`
	Gamma          float64 `json:"peakEnhancement"`
	Spread         float64 `json:"spread"`
	Swell          float64 `json:"swell"`
	SwellDirection float64 `json:"swellDirection"`
	Seed           uint32  `json:"seed"`
	WindDirection  float64 `json:"windDirection"`
}

var swellPresets = []swellPreset{
	{Name: "calm", WindSpeed: 3, Fetch: 30, Gamma: 3.3, Spread: 16, Swell: 0.1, Seed: 1, WindDirection: 0},
	{Name: "moderate", WindSpeed: 7, Fetch: 100, Gamma: 3.3, Spread: 8, Swell: 0.15, Seed: 1, WindDirection: 30},
	{Name: "rough", WindSpeed: 11, Fetch: 200, Gamma: 3.3, Spread: 6, Swell: 0.25, Seed: 1, WindDirection: 200},
	{Name: "storm", WindSpeed: 20, Fetch: 250, Gamma: 2, Spread: 4, Swell: 0.35, Seed: 1, WindDirection: 300},
}

const swellCount = 24

type swellBin struct {
	Cascade   int     `json:"cascade"`
	IX        int     `json:"ix"`
	IZ        int     `json:"iz"`
	K         float64 `json:"k"`
	Omega     float64 `json:"omega"`
	Amplitude float64 `json:"amplitude"`
	Phase     float64 `json:"phase"`
}

type swellGolden struct {
	Preset swellPreset `json:"preset"`
	Bins   []swellBin  `json:"bins"`
}

// elioDir converts a GoSX direction in degrees (0 toward +Z, positive toward
// +X) to the kernel's windDir and swellDir uniforms: the angle theta of the
// direction the waves travel toward, vector (cos theta, sin theta) in (x, z).
func elioDir(degrees float64) float64 { return math.Pi/2 - degrees*math.Pi/180 }

func (p swellPreset) spec(cascade int) oceanSpec {
	s := defaultSpec(cascade, 64)
	s.seed = int(p.Seed)
	s.windSpeed, s.fetchKm, s.gamma, s.spread, s.swell = p.WindSpeed, p.Fetch, p.Gamma, p.Spread, p.Swell
	s.windDir, s.swellDir = elioDir(p.WindDirection), elioDir(p.SwellDirection)
	return s
}

// swellBins lists every non-zero bin of one cascade as a plane wave.
func swellBins(t *testing.T, s oceanSpec) []swellBin {
	h0, omega := runOceanSpectrum(t, s)
	dk := 2 * math.Pi / s.tile
	var out []swellBin
	for z := 0; z < s.n; z++ {
		for x := 0; x < s.n; x++ {
			g := h0[z*s.n+x]
			if g[0] == 0 && g[1] == 0 {
				continue
			}
			ix, iz := signedBin(x, s.n), signedBin(z, s.n)
			out = append(out, swellBin{
				Cascade: s.cascade, IX: ix, IZ: iz,
				K:         dk * math.Sqrt(float64(ix*ix+iz*iz)),
				Omega:     omega[z*s.n+x],
				Amplitude: 2 * math.Hypot(g[0], g[1]),
				Phase:     math.Atan2(g[1], g[0]),
			})
		}
	}
	return out
}

// selectSwell keeps the count strongest bins: by amplitude, then smaller |k|,
// then smaller signed index (ix, then iz), then smaller cascade.
func selectSwell(bins []swellBin, count int) []swellBin {
	sort.SliceStable(bins, func(i, j int) bool {
		a, b := bins[i], bins[j]
		switch {
		case a.Amplitude != b.Amplitude:
			return a.Amplitude > b.Amplitude
		case a.K != b.K:
			return a.K < b.K
		case a.IX != b.IX:
			return a.IX < b.IX
		case a.IZ != b.IZ:
			return a.IZ < b.IZ
		}
		return a.Cascade < b.Cascade
	})
	if len(bins) > count {
		bins = bins[:count]
	}
	return bins
}

func buildSwellGolden(t *testing.T) []byte {
	var out []swellGolden
	for _, p := range swellPresets {
		bins := append(swellBins(t, p.spec(0)), swellBins(t, p.spec(1))...)
		out = append(out, swellGolden{Preset: p, Bins: selectSwell(bins, swellCount)})
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(data, '\n')
}

func TestOceanSwellBinsGolden(t *testing.T) {
	got := buildSwellGolden(t)
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll(filepath.Dir(swellGoldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(swellGoldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s; copy it to gosx scene/testdata/ocean-swell-bins.json", swellGoldenPath)
		return
	}
	want, err := os.ReadFile(swellGoldenPath)
	if err != nil {
		t.Fatalf("%v (run with UPDATE_GOLDEN=1)", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s is stale: run with UPDATE_GOLDEN=1, then copy it to gosx", swellGoldenPath)
	}
}

// TestOceanSwellGoldenMatchesGoSX keeps the GoSX copy byte-identical. It skips
// when the gosx sibling checkout does not have the file yet.
func TestOceanSwellGoldenMatchesGoSX(t *testing.T) {
	want, err := os.ReadFile(swellGoldenPath)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join("..", "..", "gosx", "scene", "testdata", "ocean-swell-bins.json")
	got, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("gosx sibling checkout has no %s", path)
	}
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s differs from %s: regenerate here with UPDATE_GOLDEN=1 and copy it to gosx", path, swellGoldenPath)
	}
}

// TestOceanSwellWavesSumToField checks the plane-wave identity: the direct sum
// of h(k, t) e^{i k.x} over every bin equals the sum of A cos(k.x - w t + phase).
func TestOceanSwellWavesSumToField(t *testing.T) {
	p := swellPresets[1]
	p.Swell = 0.3
	s := p.spec(1)
	h0, omega := runOceanSpectrum(t, s)
	waves := swellBins(t, s)
	dk := 2 * math.Pi / s.tile
	points := [][3]float64{{0, 0, 0}, {13.7, -41.2, 3.5}, {-250, 99, 17}, {400, 400, 600.25}}
	var worst, scale float64
	for _, pt := range points {
		x, z, tm := pt[0], pt[1], pt[2]
		var direct, plane float64
		for zi := 0; zi < s.n; zi++ {
			for xi := 0; xi < s.n; xi++ {
				g := h0[zi*s.n+xi]
				w := omega[zi*s.n+xi]
				kx, kz := dk*float64(signedBin(xi, s.n)), dk*float64(signedBin(zi, s.n))
				// h(k,t) = h0 e^{-iwt} + m e^{+iwt}; field term Re[h e^{i k.x}].
				ph := kx*x + kz*z
				direct += (g[0]*math.Cos(ph-w*tm) - g[1]*math.Sin(ph-w*tm)) +
					(g[2]*math.Cos(ph+w*tm) - g[3]*math.Sin(ph+w*tm))
			}
		}
		for _, b := range waves {
			kx, kz := dk*float64(b.IX), dk*float64(b.IZ)
			plane += b.Amplitude * math.Cos(kx*x+kz*z-b.Omega*tm+b.Phase)
		}
		// Bin k's m(k) term is the mirror bin's h0(-k) term conjugated, so every
		// bin's h0 wave, with amplitude 2|h0|, carries both halves.
		worst = math.Max(worst, math.Abs(direct-plane))
		scale = math.Max(scale, math.Abs(direct))
	}
	t.Logf("direct field vs plane-wave sum: worst difference %.3g m against field scale %.3g m", worst, scale)
	if worst > 1e-9*math.Max(scale, 1) {
		t.Errorf("plane-wave sum differs from the field by %g m (scale %g m)", worst, scale)
	}
}

// TestOceanSwellTravelsTowardAuthoredDirection checks that, with elioDir, the
// energy-weighted mean of cos(angle between the wave direction k and the
// authored direction) is well above zero for a pure wind sea (about 0.65 at spread 8).
func TestOceanSwellTravelsTowardAuthoredDirection(t *testing.T) {
	for _, deg := range []float64{0, 30, 200, 300} {
		p := swellPresets[1]
		p.Swell, p.WindDirection = 0, deg
		bins := swellBins(t, p.spec(1))
		wx, wz := math.Sin(deg*math.Pi/180), math.Cos(deg*math.Pi/180)
		var dot, total float64
		for _, b := range bins {
			n := math.Hypot(float64(b.IX), float64(b.IZ))
			dot += b.Amplitude * b.Amplitude * (float64(b.IX)*wx + float64(b.IZ)*wz) / n
			total += b.Amplitude * b.Amplitude
		}
		if c := dot / total; c < 0.5 {
			t.Errorf("wind %g deg: waves travel with mean cosine %.3f to the authored direction, want > 0.5", deg, c)
		}
	}
}
