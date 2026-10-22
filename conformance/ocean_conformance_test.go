package conformance

import (
	"fmt"
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/gather"
	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/sema"
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
		{"ocean_assemble", stdlib.OceanAssemble, []string{"kernel void ocean_assemble", "constant OceanAssembleParams&", "access::write"}},
		{"ocean_evolve", stdlib.OceanEvolve, []string{"kernel void ocean_evolve", "constant OceanEvolveParams&"}},
		{"ocean_probe", stdlib.OceanProbe, []string{"kernel void ocean_probe", "constant OceanProbeParams&", "const device float4* disp0"}},
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

// TestOceanGatherKernels runs the gather-compatible ocean kernels (and the
// gather FFT at the sizes the ocean uses) through emit/gather and checks the
// GLSL ES 3.00 output with glslangValidator. This is the permanent form of the
// scratch check from integration step I1: sema.CheckGather accepts the gather
// FFT and its fragment shader validates at n = 64 and n = 256.
func TestOceanGatherKernels(t *testing.T) {
	type tc struct {
		name    string
		build   func() (*ir.Module, error)
		kernel  string
		inputs  int
		outputs int
	}
	cases := []tc{
		{"ocean_evolve", stdlib.OceanEvolve, "ocean_evolve", 2, 2},
		{"ocean_assemble", stdlib.OceanAssemble, "ocean_assemble", 2, 2},
	}
	for _, n := range []int{64, 256} {
		n := n
		cases = append(cases, tc{fmt.Sprintf("fft_stage/n=%d", n), func() (*ir.Module, error) { return stdlib.FFT2DGather(n) }, "fft_stage", 1, 1})
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			mod, err := c.build()
			if err != nil {
				t.Fatal(err)
			}
			if errs := sema.CheckGather(mod, c.kernel); len(errs) != 0 {
				t.Fatalf("CheckGather: %v", errs)
			}
			src, lay, err := gather.Emit(mod, c.kernel)
			if err != nil {
				t.Fatalf("gather.Emit: %v", err)
			}
			if len(lay.Inputs) != c.inputs || len(lay.Outputs) != c.outputs {
				t.Errorf("layout inputs=%d outputs=%d, want %d and %d", len(lay.Inputs), len(lay.Outputs), c.inputs, c.outputs)
			}
			validateES(t, src)
		})
	}
}
