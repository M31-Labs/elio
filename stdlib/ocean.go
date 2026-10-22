package stdlib

import (
	_ "embed"
	"fmt"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/parse"
)

// OceanRngSource is the .elio source of the ocean hash RNG kernel
// (ocean_rng): PCG hash of (seed, cascade, signed ix, signed iz) to two
// uniforms in (0, 1] and two Box-Muller normals. It is a test and reference
// kernel for the block that ocean_spectrum repeats. The hash itself is spliced
// in by OceanRng (see oceanRngWords); the bare source has a placeholder.
//
//go:embed ocean/rng.elio
var OceanRngSource string

// OceanSpectrumSource is the .elio source of ocean_spectrum: JONSWAP with
// Hasselmann spreading for one cascade, written as h0 and omega buffers. The
// source header documents the model, units and conventions. The hash block is
// spliced in by OceanSpectrum.
//
//go:embed ocean/spectrum.elio
var OceanSpectrumSource string

// OceanSpectrum builds the ocean_spectrum module from OceanSpectrumSource.
func OceanSpectrum() (*ir.Module, error) {
	return parseOcean("ocean/spectrum.elio", OceanSpectrumSource, true)
}

// OceanEvolveSource is the .elio source of ocean_evolve: it evolves h0 to time
// t and writes the eight displacement and slope spectra packed as four complex
// numbers per bin in two buffers, each transformed by one inverse FFT. It is
// gather-compatible (two RGBA32F outputs).
//
//go:embed ocean/evolve.elio
var OceanEvolveSource string

// OceanEvolve builds the ocean_evolve module from OceanEvolveSource.
func OceanEvolve() (*ir.Module, error) {
	return parseOcean("ocean/evolve.elio", OceanEvolveSource, false)
}

// OceanProbeSource is the .elio source of ocean_probe: for up to 64 world
// points it sums the cascades' displacement buffers by bilinear sampling,
// inverts the horizontal displacement (two fixed-point and four Newton steps)
// and writes (height, normal.x, normal.z, t). The source header documents the
// expected buffer layout.
//
//go:embed ocean/probe.elio
var OceanProbeSource string

// OceanProbe builds the ocean_probe module from OceanProbeSource.
func OceanProbe() (*ir.Module, error) {
	return parseOcean("ocean/probe.elio", OceanProbeSource, false)
}

// OceanRng builds the ocean_rng module from OceanRngSource.
func OceanRng() (*ir.Module, error) {
	return parseOcean("ocean/rng.elio", OceanRngSource, true)
}

// oceanRngMarker is the placeholder statement `let ocean_rng_words = 0u;` that
// parseOcean replaces with oceanRngWords.
const oceanRngMarker = "ocean_rng_words"

// parseOcean parses src and, if rng is set, replaces the marker statement in
// every kernel with the hash block. It fails if the marker is missing.
func parseOcean(name, src string, rng bool) (*ir.Module, error) {
	mod, err := parse.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("stdlib: %s: %w", name, err)
	}
	if !rng {
		return mod, nil
	}
	for i := range mod.Kernels {
		body, ok := spliceMarker(mod.Kernels[i].Body, oceanRngMarker, oceanRngWords())
		if !ok {
			return nil, fmt.Errorf("stdlib: %s: kernel %q has no %s marker", name, mod.Kernels[i].Name, oceanRngMarker)
		}
		mod.Kernels[i].Body = body
	}
	return mod, nil
}

// spliceMarker replaces the first `let <marker> = ...` statement found in
// stmts (searching into if and loop bodies) with repl.
func spliceMarker(stmts []ir.Stmt, marker string, repl []ir.Stmt) ([]ir.Stmt, bool) {
	out := make([]ir.Stmt, 0, len(stmts)+len(repl))
	found := false
	for _, s := range stmts {
		if found {
			out = append(out, s)
			continue
		}
		switch x := s.(type) {
		case ir.Let:
			if x.Name == marker {
				out = append(out, repl...)
				found = true
				continue
			}
		case ir.If:
			if then, ok := spliceMarker(x.Then, marker, repl); ok {
				x.Then, found = then, true
			} else if els, ok := spliceMarker(x.Else, marker, repl); ok {
				x.Else, found = els, true
			}
			s = x
		case ir.For:
			if body, ok := spliceMarker(x.Body, marker, repl); ok {
				x.Body, found = body, true
			}
			s = x
		case ir.While:
			if body, ok := spliceMarker(x.Body, marker, repl); ok {
				x.Body, found = body, true
			}
			s = x
		}
		out = append(out, s)
	}
	return out, found
}

// oceanRngWords is the PCG hash block of the ocean RNG. It reads the u32 names
// rseed and rcascade and the i32 names rix and riz, and assigns the u32
// variables rb1 and rb2 (declared by the kernel).
//
// PCG output hash (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// 2020), absorbing six words in turn: seed, cascade, ix, iz (signed, as two's
// complement), then two salts that split the key hash into two uniform streams
// (rb1 after word 4, rb2 after word 5). Each step is
//
//	s = (h ^ w) * 747796405 + 2891336453
//	v = ((s >> ((s >> 28) + 4)) ^ s) * 277803737
//	h = (v >> 22) ^ v
//
// The key is the signed wavenumber index, not the texel, so a resolution
// change never reshuffles a wave. The interpreter holds u32 in int64, so the
// sum is masked to 32 bits (a no-op on a GPU); products already wrap there.
// No literal-by-literal product appears, which WGSL would fold at compile time.
func oceanRngWords() []ir.Stmt {
	u := func(s string) ir.Expr { return ir.Lit{Text: s + "u"} }
	n := func(s string) ir.Expr { return ir.Name{Name: s} }
	b := func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	eqR := func(k string) ir.Expr { return b("==", n("rr"), u(k)) }
	pick := func(k, w string, v ir.Expr) ir.Stmt {
		return ir.If{Cond: eqR(k), Then: []ir.Stmt{ir.Assign{Target: n(w), Value: v}}}
	}
	mask := func(e ir.Expr) ir.Expr { return b("&", e, u("4294967295")) }
	cast := func(e ir.Expr) ir.Expr { return ir.Call{Func: "u32", Args: []ir.Expr{e}} }
	return []ir.Stmt{
		ir.Var{Name: "rh", Type: ir.U32, Init: u("0")},
		ir.For{
			Init: ir.Var{Name: "rr", Type: ir.U32, Init: u("0")},
			Cond: b("<", n("rr"), u("6")),
			Post: ir.Assign{Target: n("rr"), Value: b("+", n("rr"), u("1"))},
			Body: []ir.Stmt{
				ir.Var{Name: "rw", Type: ir.U32, Init: n("rseed")},
				pick("1", "rw", n("rcascade")),
				pick("2", "rw", mask(cast(n("rix")))),
				pick("3", "rw", mask(cast(n("riz")))),
				pick("4", "rw", u("2654435769")),
				pick("5", "rw", u("2246822519")),
				ir.Let{Name: "rs", Value: mask(b("+", b("*", b("^", n("rh"), n("rw")), u("747796405")), u("2891336453")))},
				ir.Let{Name: "rv", Value: b("*", b("^", b(">>", n("rs"), b("+", b(">>", n("rs"), u("28")), u("4"))), n("rs")), u("277803737"))},
				ir.Assign{Target: n("rh"), Value: b("^", b(">>", n("rv"), u("22")), n("rv"))},
				pick("4", "rb1", n("rh")),
				pick("5", "rb2", n("rh")),
			},
		},
	}
}
