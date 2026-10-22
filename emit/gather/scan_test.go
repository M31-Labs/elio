package gather_test

import (
	"strings"
	"testing"

	"m31labs.dev/elio/emit/gather"
	"m31labs.dev/elio/stdlib"
)

func TestEmitRejectsNonGather(t *testing.T) {
	_, _, err := gather.Emit(stdlib.Scan(), "scan")
	if err == nil || !strings.Contains(err.Error(), "rule 1") {
		t.Fatalf("want a gather profile error, got %v", err)
	}
}
