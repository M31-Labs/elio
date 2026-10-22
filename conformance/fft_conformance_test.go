package conformance

import (
	"fmt"
	"strings"
	"testing"

	prismvalidate "m31labs.dev/prism/validate"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/stdlib"
)

func TestFFTAllBackends(t *testing.T) {
	forms := []struct {
		name  string
		build func(int) (*ir.Module, error)
		metal []string
	}{
		{"compute", stdlib.FFT2DCompute, []string{"threadgroup", "threadgroup_barrier"}},
		{"gather", stdlib.FFT2DGather, []string{"kernel void fft_stage", "constant FFTStage&"}},
	}
	for _, f := range forms {
		for _, n := range []int{64, 256} {
			t.Run(fmt.Sprintf("%s/n=%d", f.name, n), func(t *testing.T) {
				mod, err := f.build(n)
				if err != nil {
					t.Fatal(err)
				}
				wsrc, err := wgsl.Emit(mod)
				if err != nil {
					t.Fatalf("wgsl.Emit: %v", err)
				}
				prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(p string) []string { return []string{p} })

				for _, k := range mod.Kernels {
					// GLSL emits one entry point per module; validate each kernel on its own.
					one := &ir.Module{Structs: mod.Structs, Consts: mod.Consts, Bindings: mod.Bindings, Kernels: []ir.Kernel{k}}
					gsrc, err := glsl.Emit(one)
					if err != nil {
						t.Fatalf("glsl.Emit %s: %v", k.Name, err)
					}
					prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(p string) []string { return []string{"-V", p, "-S", "comp"} })
				}

				msrc, err := metal.Emit(mod)
				if err != nil {
					t.Fatalf("metal.Emit: %v", err)
				}
				for _, want := range f.metal {
					if !strings.Contains(msrc, want) {
						t.Errorf("metal missing %q\n%s", want, msrc)
					}
				}
			})
		}
	}
}
