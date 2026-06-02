package gosx

import (
	"strings"
	"testing"

	"m31labs.dev/elio/ir"
	"m31labs.dev/gosx/render/compute"
	"m31labs.dev/gosx/render/gpu"
	"m31labs.dev/gosx/render/gpu/headless"
)

// TestCullPassRecordsAndPublishes drives the flagship frustum-cull kernel through
// the adapter: New compiles it to WGSL, and Record builds the pipeline, binds the
// kernel's four buffers from their Elio binding names, dispatches, and publishes
// the compacted-instance + indirect-args buffers onto the bus — the exact shape
// the renderer's main pass consumes. It proves an Elio kernel becomes a working
// render/compute.ExternalComputePass with no hand-written GPU plumbing.
func TestCullPassRecordsAndPublishes(t *testing.T) {
	dev, _ := headless.New(64, 64)
	mod := ir.CullKernel() // bindings: cull (uniform), input, output, drawArgs

	bufs := map[string]gpu.Buffer{}
	mk := func(name string, size int, usage gpu.BufferUsage) gpu.Buffer {
		b, err := dev.CreateBuffer(gpu.BufferDesc{Size: size, Label: name, Usage: usage})
		if err != nil {
			t.Fatalf("buffer %s: %v", name, err)
		}
		bufs[name] = b
		return b
	}
	mk("cull", 64, gpu.BufferUsageUniform|gpu.BufferUsageCopyDst)
	mk("input", compute.InstanceRecordStride*4, gpu.BufferUsageStorage|gpu.BufferUsageCopyDst)
	out := mk("output", compute.InstanceRecordStride*4, gpu.BufferUsageStorage|gpu.BufferUsageVertex|gpu.BufferUsageCopyDst)
	args := mk("drawArgs", compute.IndirectArgsStride, gpu.BufferUsageStorage|gpu.BufferUsageIndirect|gpu.BufferUsageCopyDst)

	pass, err := New(Config{
		ID:      "elio.cull",
		Phase:   compute.PhaseAfterCull,
		Module:  mod,
		Kernel:  "main",
		Buffers: func(n string) (gpu.Buffer, bool) { b, ok := bufs[n]; return b, ok },
		Groups:  func() (int, int, int) { return 1, 1, 1 },
		Publish: []compute.GPUResource{
			{Name: "hero.instances", Buffer: out, Role: compute.RoleInstanceAttr, Element: compute.InstanceRecordLayout(), Count: 4, Access: compute.Read},
			{Name: "hero.drawArgs", Buffer: args, Role: compute.RoleIndirectArgs, Element: compute.IndirectArgsLayout(), Count: 1, Access: compute.Read},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if pass.ID() != "elio.cull" || pass.Phase() != compute.PhaseAfterCull {
		t.Fatalf("ID/Phase = %q/%v", pass.ID(), pass.Phase())
	}
	if !strings.Contains(pass.WGSL(), "fn main(") {
		t.Errorf("compiled WGSL missing entry point:\n%s", pass.WGSL())
	}

	var published []compute.GPUResource
	enc := dev.CreateCommandEncoder()
	if err := pass.Record(compute.PassContext{
		Device:  dev,
		Encoder: enc,
		Frame:   1,
		Publish: func(r compute.GPUResource) { published = append(published, r) },
	}); err != nil {
		t.Fatalf("Record: %v", err)
	}

	if len(published) != 2 {
		t.Fatalf("published %d resources, want 2", len(published))
	}
	if published[0].Name != "hero.instances" || published[0].Role != compute.RoleInstanceAttr {
		t.Errorf("published[0] = %+v", published[0])
	}
	if published[1].Name != "hero.drawArgs" || published[1].Role != compute.RoleIndirectArgs {
		t.Errorf("published[1] = %+v", published[1])
	}

	// Recording again must reuse the pipeline (lazy build happens once).
	enc2 := dev.CreateCommandEncoder()
	if err := pass.Record(compute.PassContext{Device: dev, Encoder: enc2, Frame: 2, Publish: func(compute.GPUResource) {}}); err != nil {
		t.Fatalf("second Record: %v", err)
	}
}

// TestMissingBufferIsReported ensures an unbound binding is a clear error, not a
// panic or a silently-wrong dispatch.
func TestMissingBufferIsReported(t *testing.T) {
	dev, _ := headless.New(8, 8)
	pass, err := New(Config{
		ID: "elio.cull", Phase: compute.PhaseAfterCull, Module: ir.CullKernel(), Kernel: "main",
		Buffers: func(string) (gpu.Buffer, bool) { return nil, false }, // resolves nothing
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	enc := dev.CreateCommandEncoder()
	err = pass.Record(compute.PassContext{Device: dev, Encoder: enc, Publish: func(compute.GPUResource) {}})
	if err == nil || !strings.Contains(err.Error(), "no buffer bound for") {
		t.Fatalf("expected unbound-binding error, got: %v", err)
	}
}
