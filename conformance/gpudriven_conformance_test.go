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

// TestGPUDrivenKernelsAllBackends emits the Scene3D GPU-driven kernels for
// every text backend and validates each with its external validator when the
// tool is on PATH. CPU execution lives in stdlib/gpudriven_test.go.
func TestGPUDrivenKernelsAllBackends(t *testing.T) {
	for _, tc := range []struct {
		name   string
		kernel string
		build  func() (*ir.Module, error)
	}{
		{"cull", "cull", stdlib.GPUDrivenCull},
		{"hiz_downsample", "downsample", stdlib.GPUDrivenHiZDownsample},
	} {
		mod, err := tc.build()
		if err != nil {
			t.Fatal(err)
		}
		wsrc, err := wgsl.Emit(mod)
		if err != nil {
			t.Fatalf("%s wgsl: %v", tc.name, err)
		}
		if !strings.Contains(wsrc, "fn "+tc.kernel+"(") {
			t.Errorf("%s wgsl lacks entry %q", tc.name, tc.kernel)
		}
		gsrc, err := glsl.Emit(mod)
		if err != nil {
			t.Fatalf("%s glsl: %v", tc.name, err)
		}
		msrc, err := metal.Emit(mod)
		if err != nil {
			t.Fatalf("%s metal: %v", tc.name, err)
		}
		if !strings.Contains(msrc, "kernel void "+tc.kernel+"(") {
			t.Errorf("%s metal lacks kernel %q\n%s", tc.name, tc.kernel, msrc)
		}
		// External validators skip their own subtest when the tool is absent,
		// so the emission checks above always run.
		t.Run(tc.name+"/naga", func(t *testing.T) {
			prismvalidate.Shader(t, "naga", wsrc, ".wgsl", func(f string) []string { return []string{f} })
		})
		t.Run(tc.name+"/glslang", func(t *testing.T) {
			prismvalidate.Shader(t, "glslangValidator", gsrc, ".comp", func(f string) []string { return []string{"-V", f, "-S", "comp"} })
		})
	}
}
