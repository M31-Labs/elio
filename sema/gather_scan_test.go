package sema_test

import (
	"strings"
	"testing"

	"m31labs.dev/elio/sema"
	"m31labs.dev/elio/stdlib"
)

func hasDiag(errs []error, subs ...string) bool {
	for _, e := range errs {
		ok := true
		for _, s := range subs {
			if !strings.Contains(e.Error(), s) {
				ok = false
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func TestCheckGatherScanFails(t *testing.T) {
	errs := sema.CheckGather(stdlib.Scan(), "scan")
	if !hasDiag(errs, "rule 1", "shared") {
		t.Errorf("missing shared-memory diagnostic: %v", errs)
	}
	if !hasDiag(errs, "rule 1", "barrier") {
		t.Errorf("missing barrier diagnostic: %v", errs)
	}
	if !hasDiag(errs, "rule 2", "local_invocation_id") {
		t.Errorf("missing builtin diagnostic: %v", errs)
	}
	for _, e := range errs {
		if !strings.Contains(e.Error(), `"scan"`) {
			t.Errorf("diagnostic does not name the kernel: %v", e)
		}
	}
}
