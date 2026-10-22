package stdlib

import (
	"regexp"
	"strings"
	"testing"

	"m31labs.dev/elio/emit/ts"
	prismvalidate "m31labs.dev/prism/validate"
)

func TestOceanKernelSetNames(t *testing.T) {
	set, err := OceanKernelSet()
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	gather := map[string]bool{}
	for _, k := range set.Kernels {
		got = append(got, k.Name)
		gather[k.Name] = k.Gather
	}
	want := []string{"ocean_spectrum", "ocean_evolve", "ocean_assemble", "ocean_mip", "ocean_foam", "ocean_probe",
		"fft_compute_64", "fft_gather_64", "fft_compute_128", "fft_gather_128", "fft_compute_256", "fft_gather_256"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("kernels = %v", got)
	}
	for _, n := range []string{"ocean_assemble", "ocean_mip", "ocean_foam", "ocean_evolve", "fft_gather_64"} {
		if !gather[n] {
			t.Errorf("%s should be a gather kernel", n)
		}
	}
	if gather["fft_compute_64"] || gather["ocean_spectrum"] || gather["ocean_probe"] {
		t.Error("compute-only kernel marked gather")
	}
}

var tsString = regexp.MustCompile("(?s)export const (\\w+(?:WGSL|GLSL)) = `(.*?[^\\\\])`;\n")

// TestOceanModuleShadersValidate extracts every shader from the generated
// module and runs naga on WGSL and glslangValidator on GLSL.
func TestOceanModuleShadersValidate(t *testing.T) {
	set, err := OceanKernelSet()
	if err != nil {
		t.Fatal(err)
	}
	out, err := ts.Emit(set, ts.Meta{Version: "test"})
	if err != nil {
		t.Fatal(err)
	}
	found := tsString.FindAllStringSubmatch(out, -1)
	if len(found) < 12 {
		t.Fatalf("found only %d shaders", len(found))
	}
	for _, m := range found {
		name, src := m[1], ts.UnescapeTemplate(m[2])
		t.Run(name, func(t *testing.T) {
			if strings.HasSuffix(name, "WGSL") {
				prismvalidate.Shader(t, "naga", src, ".wgsl", func(p string) []string { return []string{p} })
			} else {
				prismvalidate.Shader(t, "glslangValidator", src, ".frag", func(p string) []string { return []string{"-S", "frag", p} })
			}
		})
	}
}
