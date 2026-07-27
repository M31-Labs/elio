package conformance

import (
	"math"
	"regexp"
	"strconv"
	"testing"

	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/stdlib"
)

// u32LitMulRe finds every `<u32 literal> * <u32 literal>` pair in emitted
// WGSL text (decimal or hex, either operand order, tolerant of the
// surrounding whitespace the emitter happens to use). Both operands must
// carry the `u` suffix — that is what makes the multiply a const-expression
// candidate; a bare `f32`/`i32` literal or a variable name never matches.
var u32LitMulRe = regexp.MustCompile(`(?i)\b(0x[0-9a-f]+|[0-9]+)u\s*\*\s*(0x[0-9a-f]+|[0-9]+)u\b`)

// checkNoU32ConstMultiplyOverflow scans WGSL source for literal*literal u32
// multiplications whose product exceeds u32::MAX (2^32-1).
//
// WGSL requires const-expressions — both operands compile-time literal — to
// evaluate without overflow: overflow there is a hard shader-creation error.
// A `variable * literal` (or `variable * variable`) multiply is a runtime
// operation that is allowed to wrap silently, so this check only inspects
// literal*literal pairs and never flags those.
//
// This is the regression guard for the class of bug fixed in
// GalaxyParticleSimulate's hash2 (stdlib/galaxy_browser.go): Firefox's naga
// rejects a `10u * 3812015801u` const-expression that overflows u32, while
// the naga CLI and Chrome's Tint both accept it silently. Wiring a real
// browser (headless Firefox + WebGPU createShaderModule/getCompilationInfo)
// into this suite is the stronger check but adds a Node/Playwright/browser
// dependency to `go test`; this source-level check needs neither a browser
// nor a network connection and would have caught the original defect before
// it ever reached naga or a browser.
func checkNoU32ConstMultiplyOverflow(t *testing.T, label, wsrc string) {
	t.Helper()
	for _, m := range u32LitMulRe.FindAllStringSubmatch(wsrc, -1) {
		a := parseU32Lit(t, m[1])
		b := parseU32Lit(t, m[2])
		product := a * b // uint64: no wraparound here, we want the true product
		if product > math.MaxUint32 {
			t.Errorf("%s: const-expression multiply %su * %su = %d overflows u32 (max %d); "+
				"Firefox's naga rejects this at shader-creation time even though the naga CLI "+
				"and Chrome's Tint accept it — fold this literal*literal product in Go with "+
				"explicit u32 wrapping before emitting it",
				label, m[1], m[2], product, uint32(math.MaxUint32))
		}
	}
}

func parseU32Lit(t *testing.T, s string) uint64 {
	t.Helper()
	v, err := strconv.ParseUint(s, 0, 64)
	if err != nil {
		t.Fatalf("parseU32Lit(%q): %v", s, err)
	}
	return v
}

// TestNoU32ConstMultiplyOverflowInEmittedWGSL runs checkNoU32ConstMultiplyOverflow
// against the WGSL emitted for every stdlib kernel, so a future hash/mix
// constant (or any other literal*literal u32 multiply) that overflows u32
// fails `go test` immediately instead of surfacing only in Firefox.
func TestNoU32ConstMultiplyOverflowInEmittedWGSL(t *testing.T) {
	kernels := map[string]*ir.Module{
		"GalaxyParticleSimulate": stdlib.GalaxyParticleSimulate(),
		"GalaxyParticleUpdate":   stdlib.GalaxyParticleUpdate(),
		"ParticleUpdate":         stdlib.ParticleUpdate(),
		"HiZDownsample":          stdlib.HiZDownsample(),
		"Skin":                   stdlib.Skin(),
		"Reduce":                 stdlib.Reduce(),
		"Scan":                   stdlib.Scan(),
		"Sort":                   stdlib.Sort(),
		"Compact":                stdlib.Compact(),
	}
	for name, mod := range kernels {
		t.Run(name, func(t *testing.T) {
			wsrc, err := wgsl.Emit(mod)
			if err != nil {
				t.Fatalf("wgsl.Emit(%s): %v", name, err)
			}
			checkNoU32ConstMultiplyOverflow(t, name, wsrc)
		})
	}
}

// TestGalaxyParticleSimulateHash2FoldedConstantsMatchWrappedProduct pins the
// exact folded values GalaxyParticleSimulate must emit for every b constant
// hash2 is called with, computed independently here via Go's defined-wrapping
// uint32 multiplication (which is bit-identical to what a GPU computes for a
// runtime u32 multiply, and to what a const-evaluator computes once told to
// wrap). If stdlib/galaxy_browser.go ever changes these constants (e.g. a new
// hash2(index, Nu) call site, or the multiplier itself), this test must be
// updated in lock-step — that coupling is intentional.
func TestGalaxyParticleSimulateHash2FoldedConstantsMatchWrappedProduct(t *testing.T) {
	const multiplier = 3812015801
	bs := []uint32{0, 1, 2, 10, 11, 12, 20, 21, 30, 31, 32, 33, 44, 46, 47, 90, 91, 92, 93}

	wsrc, err := wgsl.Emit(stdlib.GalaxyParticleSimulate())
	if err != nil {
		t.Fatalf("wgsl.Emit: %v", err)
	}

	for _, b := range bs {
		want := b * multiplier // uint32 wraps automatically; identical to GPU runtime wrap
		wantText := strconv.FormatUint(uint64(want), 10) + "u"
		if !regexp.MustCompile(`\b` + wantText + `\b`).MatchString(wsrc) {
			t.Errorf("folded hash2(_, %du) constant %s not found in emitted WGSL; "+
				"want b*%du wrapped as u32 = %s", b, wantText, multiplier, wantText)
		}
		// The un-folded literal*literal form must never appear again.
		badText := strconv.FormatUint(uint64(b), 10) + "u * " + strconv.FormatUint(multiplier, 10) + "u"
		if regexp.MustCompile(regexp.QuoteMeta(badText)).MatchString(wsrc) {
			t.Errorf("unfolded const-expression %q still present in emitted WGSL", badText)
		}
	}
}
