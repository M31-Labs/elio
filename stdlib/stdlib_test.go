package stdlib

import (
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// TestScanIsValid pins that the scan kernel passes semantic analysis.
func TestScanIsValid(t *testing.T) {
	if errs := sema.Check(Scan()); len(errs) != 0 {
		t.Fatalf("scan failed sema:\n%v", sema.Errors(errs))
	}
}

// TestScanComputesPrefixSums executes the scan on the lockstep CPU interpreter
// over a 64-lane tile of ones and asserts an inclusive prefix sum: output[i] =
// i+1. This proves the cooperative algorithm — shared memory, the log-step loop,
// and its barriers — is correct, not merely that it emits valid shader source.
func TestScanComputesPrefixSums(t *testing.T) {
	const n = 64
	input := make([]float64, n)
	for i := range input {
		input[i] = 1
	}
	output := make([]float64, n)
	mem := &run.Memory{Vars: map[string]any{"input": input, "output": output}}

	if err := run.Run(Scan(), "scan", n, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for i := 0; i < n; i++ {
		if want := float64(i + 1); output[i] != want {
			t.Fatalf("output[%d] = %v, want %v (inclusive prefix sum of ones)", i, output[i], want)
		}
	}
}

// TestScanRamp scans 0,1,2,…,63 and checks the triangular prefix sums
// output[i] = i*(i+1)/2, exercising non-uniform input.
func TestScanRamp(t *testing.T) {
	const n = 64
	input := make([]float64, n)
	for i := range input {
		input[i] = float64(i)
	}
	output := make([]float64, n)
	mem := &run.Memory{Vars: map[string]any{"input": input, "output": output}}

	if err := run.Run(Scan(), "scan", n, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for i := 0; i < n; i++ {
		if want := float64(i * (i + 1) / 2); output[i] != want {
			t.Fatalf("output[%d] = %v, want %v", i, output[i], want)
		}
	}
}
