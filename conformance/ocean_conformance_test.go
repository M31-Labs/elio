package conformance

import (
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/stdlib"
)

// oceanKernel is one ocean kernel module and the Metal text it must contain.
type oceanKernel struct {
	name  string
	build func() (*ir.Module, error)
	metal []string
}

// oceanKernels lists every ocean kernel built from .elio or hand-built IR.
func oceanKernels() []oceanKernel {
	return []oceanKernel{
		{"ocean_rng", stdlib.OceanRng, []string{"kernel void ocean_rng", "constant OceanRngParams&"}},
		{"ocean_spectrum", stdlib.OceanSpectrum, []string{"kernel void ocean_spectrum", "constant OceanSpectrumParams&"}},
	}
}

// TestOceanKernelsAllBackends checks WGSL with naga, GLSL 450 with
// glslangValidator, and the Metal text structure for each ocean kernel.
func TestOceanKernelsAllBackends(t *testing.T) {
	for _, k := range oceanKernels() {
		t.Run(k.name, func(t *testing.T) {
			mod, err := k.build()
			if err != nil {
				t.Fatal(err)
			}
			wsrc, err := wgsl.Emit(mod)
			if err != nil {
				t.Fatalf("wgsl.Emit: %v", err)
			}
			checkNoU32ConstMultiplyOverflow(t, k.name, wsrc)
			prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(p string) []string { return []string{p} })

			for _, kn := range mod.Kernels {
				one := &ir.Module{Structs: mod.Structs, Consts: mod.Consts, Bindings: mod.Bindings, Kernels: []ir.Kernel{kn}}
				gsrc, err := glsl.Emit(one)
				if err != nil {
					t.Fatalf("glsl.Emit %s: %v", kn.Name, err)
				}
				prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(p string) []string { return []string{"-V", p, "-S", "comp"} })
			}

			msrc, err := metal.Emit(mod)
			if err != nil {
				t.Fatalf("metal.Emit: %v", err)
			}
			for _, want := range k.metal {
				if !strings.Contains(msrc, want) {
					t.Errorf("metal missing %q\n%s", want, msrc)
				}
			}
		})
	}
}
