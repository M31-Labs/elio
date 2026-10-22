// Package kernelsets groups Elio kernels into the sets a host consumes as one
// generated module (`elio emit ts --set <name>`). It sits above stdlib, the
// emitters and sema, so stdlib stays a library of kernels with no emitter
// dependencies.
package kernelsets

import (
	"errors"
	"fmt"

	"m31labs.dev/elio/emit/ts"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/sema"
	"m31labs.dev/elio/stdlib"
)

// Ocean lists the kernels the browser host needs, for `elio emit ts
// --set ocean`: the six ocean kernels, then the FFT compute and gather forms
// at n = 64, 128 and 256 (fft_compute_N, fft_gather_N). Kernels that pass
// sema.CheckGather are marked Gather; a kernel that is meant to be gather but
// fails the check fails the build here.
func Ocean() (ts.KernelSet, error) {
	set := ts.KernelSet{Name: "ocean"}
	for _, k := range []struct {
		name, entry string
		build       func() (*ir.Module, error)
		gather      bool
	}{
		{"ocean_spectrum", "ocean_spectrum", stdlib.OceanSpectrum, true},
		{"ocean_evolve", "ocean_evolve", stdlib.OceanEvolve, true},
		{"ocean_assemble", "ocean_assemble", stdlib.OceanAssemble, true},
		{"ocean_mip", "ocean_mip", stdlib.OceanMip, true},
		{"ocean_foam", "ocean_foam", stdlib.OceanFoam, true},
		{"ocean_probe", "ocean_probe", stdlib.OceanProbe, false},
	} {
		m, err := k.build()
		if err != nil {
			return ts.KernelSet{}, err
		}
		if k.gather {
			if errs := sema.CheckGather(m, k.entry); len(errs) > 0 {
				return ts.KernelSet{}, fmt.Errorf("kernelsets: %s is not gather-compatible: %w", k.name, errors.Join(errs...))
			}
		}
		set.Kernels = append(set.Kernels, ts.Kernel{Name: k.name, Entry: k.entry, Module: m, Gather: k.gather})
	}
	for _, n := range []int{64, 128, 256} {
		c, err := stdlib.FFT2DCompute(n)
		if err != nil {
			return ts.KernelSet{}, err
		}
		g, err := stdlib.FFT2DGather(n)
		if err != nil {
			return ts.KernelSet{}, err
		}
		if errs := sema.CheckGather(g, "fft_stage"); len(errs) > 0 {
			return ts.KernelSet{}, fmt.Errorf("kernelsets: fft_gather_%d is not gather-compatible: %w", n, errors.Join(errs...))
		}
		set.Kernels = append(set.Kernels,
			ts.Kernel{Name: fmt.Sprintf("fft_compute_%d", n), Entry: "fft_rows", Module: c},
			ts.Kernel{Name: fmt.Sprintf("fft_gather_%d", n), Entry: "fft_stage", Module: g, Gather: true})
	}
	return set, nil
}
