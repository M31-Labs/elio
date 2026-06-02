package stdlib

import (
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// TestPrimitivesAreValid pins that every stdlib kernel passes semantic analysis.
func TestPrimitivesAreValid(t *testing.T) {
	for _, tc := range []struct {
		name string
		errs []error
	}{
		{"Scan", sema.Check(Scan())},
		{"Compact", sema.Check(Compact())},
	} {
		if len(tc.errs) != 0 {
			t.Errorf("%s failed sema:\n%v", tc.name, sema.Errors(tc.errs))
		}
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

// TestCompactPacksSurvivors runs stream compaction over a 64-lane tile whose
// even lanes carry values 1..32 and odd lanes carry zero, asserting the
// survivors are densely packed in stable order (output[k] = k+1) and the count
// is 32 — the scan-driven, atomics-free equivalent of the cull kernel.
func TestCompactPacksSurvivors(t *testing.T) {
	const n = 64
	input := make([]float64, n)
	for i := 0; i < n; i += 2 {
		input[i] = float64(i/2 + 1) // 1,2,3,…,32 at even lanes; zero at odd
	}
	output := make([]float64, n)
	count := make([]float64, 1)
	mem := &run.Memory{Vars: map[string]any{"input": input, "output": output, "count": count}}

	if err := run.Run(Compact(), "compact", n, mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if count[0] != 32 {
		t.Fatalf("count = %v, want 32", count[0])
	}
	for k := 0; k < 32; k++ {
		if want := float64(k + 1); output[k] != want {
			t.Fatalf("output[%d] = %v, want %v", k, output[k], want)
		}
	}
}
