package stdlib

import (
	"errors"
	"testing"

	"m31labs.dev/elio/sema"
)

// WebGL2 runs ocean_spectrum as a gather pass: h0 and omega at its own index,
// two outputs of 16 bytes each.
func TestOceanSpectrumIsGatherCompatible(t *testing.T) {
	m, err := OceanSpectrum()
	if err != nil {
		t.Fatal(err)
	}
	if errs := sema.CheckGather(m, "ocean_spectrum"); len(errs) > 0 {
		t.Fatal(errors.Join(errs...))
	}
}
