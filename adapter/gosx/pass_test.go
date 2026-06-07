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

// recordingEncoder wraps a real CommandEncoder so a test can observe which
// dispatch path the adapter took. Every method except BeginComputePass delegates
// to the embedded encoder, so the headless backend still runs normally.
type recordingEncoder struct {
	gpu.CommandEncoder
	pass *recordingComputePass
}

func (e *recordingEncoder) BeginComputePass() gpu.ComputePassEncoder {
	e.pass = &recordingComputePass{inner: e.CommandEncoder.BeginComputePass()}
	return e.pass
}

// recordingComputePass records dispatch calls then delegates to the real pass.
type recordingComputePass struct {
	inner          gpu.ComputePassEncoder
	directCalls    int
	indirectCalls  int
	indirectBuffer gpu.Buffer
	indirectOffset int
}

func (p *recordingComputePass) SetPipeline(pl gpu.ComputePipeline)   { p.inner.SetPipeline(pl) }
func (p *recordingComputePass) SetBindGroup(g int, bg gpu.BindGroup) { p.inner.SetBindGroup(g, bg) }
func (p *recordingComputePass) DispatchWorkgroups(x, y, z int) {
	p.directCalls++
	p.inner.DispatchWorkgroups(x, y, z)
}
func (p *recordingComputePass) DispatchWorkgroupsIndirect(buf gpu.Buffer, off int) {
	p.indirectCalls++
	p.indirectBuffer = buf
	p.indirectOffset = off
	p.inner.DispatchWorkgroupsIndirect(buf, off)
}
func (p *recordingComputePass) End() { p.inner.End() }

// cullBufs allocates the four buffers the cull kernel binds, keyed by Elio
// binding name, on the given headless device.
func cullBufs(t *testing.T, dev gpu.Device) map[string]gpu.Buffer {
	t.Helper()
	bufs := map[string]gpu.Buffer{}
	mk := func(name string, size int, usage gpu.BufferUsage) {
		b, err := dev.CreateBuffer(gpu.BufferDesc{Size: size, Label: name, Usage: usage})
		if err != nil {
			t.Fatalf("buffer %s: %v", name, err)
		}
		bufs[name] = b
	}
	mk("cull", 64, gpu.BufferUsageUniform|gpu.BufferUsageCopyDst)
	mk("input", compute.InstanceRecordStride*4, gpu.BufferUsageStorage|gpu.BufferUsageCopyDst)
	mk("output", compute.InstanceRecordStride*4, gpu.BufferUsageStorage|gpu.BufferUsageVertex|gpu.BufferUsageCopyDst)
	mk("drawArgs", compute.IndirectArgsStride, gpu.BufferUsageStorage|gpu.BufferUsageIndirect|gpu.BufferUsageCopyDst)
	return bufs
}

// TestIndirectDispatchUsesGPUDrivenCount proves that when Config.IndirectArgs is
// set, the adapter dispatches via DispatchWorkgroupsIndirect against the supplied
// buffer+offset and never falls back to a CPU-counted DispatchWorkgroups — the
// compute-drives-compute / compute-drives-draw seam where the workgroup count
// stays on the GPU (e.g. written by a prior cull/compaction pass).
func TestIndirectDispatchUsesGPUDrivenCount(t *testing.T) {
	dev, _ := headless.New(64, 64)
	bufs := cullBufs(t, dev)

	// A separate indirect-args buffer holding the 3×u32 workgroup count.
	indirect, err := dev.CreateBuffer(gpu.BufferDesc{
		Size: 16, Label: "dispatchArgs",
		Usage: gpu.BufferUsageIndirect | gpu.BufferUsageStorage | gpu.BufferUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("indirect buffer: %v", err)
	}
	const wantOffset = 0

	groupsCalled := false
	pass, err := New(Config{
		ID: "elio.cull", Phase: compute.PhaseAfterCull, Module: ir.CullKernel(), Kernel: "main",
		Buffers:      func(n string) (gpu.Buffer, bool) { b, ok := bufs[n]; return b, ok },
		Groups:       func() (int, int, int) { groupsCalled = true; return 1, 1, 1 },
		IndirectArgs: func() (gpu.Buffer, int, bool) { return indirect, wantOffset, true },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	enc := &recordingEncoder{CommandEncoder: dev.CreateCommandEncoder()}
	if err := pass.Record(compute.PassContext{
		Device: dev, Encoder: enc, Frame: 1, Publish: func(compute.GPUResource) {},
	}); err != nil {
		t.Fatalf("Record: %v", err)
	}

	rp := enc.pass
	if rp == nil {
		t.Fatal("no compute pass was begun")
	}
	if rp.indirectCalls != 1 {
		t.Errorf("DispatchWorkgroupsIndirect called %d times, want 1", rp.indirectCalls)
	}
	if rp.directCalls != 0 {
		t.Errorf("DispatchWorkgroups (CPU-counted) called %d times, want 0", rp.directCalls)
	}
	if rp.indirectBuffer != indirect {
		t.Errorf("indirect dispatch bound the wrong buffer")
	}
	if rp.indirectOffset != wantOffset {
		t.Errorf("indirect offset = %d, want %d", rp.indirectOffset, wantOffset)
	}
	if groupsCalled {
		t.Errorf("Groups was consulted even though IndirectArgs supplied the count")
	}
}

// TestIndirectArgsOptOutFallsBackToGroups proves the seam is opt-in per frame:
// when IndirectArgs returns ok=false, the adapter uses the CPU-supplied Groups.
func TestIndirectArgsOptOutFallsBackToGroups(t *testing.T) {
	dev, _ := headless.New(64, 64)
	bufs := cullBufs(t, dev)
	pass, err := New(Config{
		ID: "elio.cull", Phase: compute.PhaseAfterCull, Module: ir.CullKernel(), Kernel: "main",
		Buffers:      func(n string) (gpu.Buffer, bool) { b, ok := bufs[n]; return b, ok },
		Groups:       func() (int, int, int) { return 2, 1, 1 },
		IndirectArgs: func() (gpu.Buffer, int, bool) { return nil, 0, false }, // opt out this frame
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	enc := &recordingEncoder{CommandEncoder: dev.CreateCommandEncoder()}
	if err := pass.Record(compute.PassContext{Device: dev, Encoder: enc, Frame: 1, Publish: func(compute.GPUResource) {}}); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if rp := enc.pass; rp.directCalls != 1 || rp.indirectCalls != 0 {
		t.Errorf("dispatch path: direct=%d indirect=%d, want direct=1 indirect=0", rp.directCalls, rp.indirectCalls)
	}
}

// TestIndirectArgsNilBufferIsReported ensures a misconfigured IndirectArgs
// (ok=true with a nil buffer) is a clear error rather than a nil deref at
// dispatch time.
func TestIndirectArgsNilBufferIsReported(t *testing.T) {
	dev, _ := headless.New(8, 8)
	bufs := cullBufs(t, dev)
	pass, err := New(Config{
		ID: "elio.cull", Phase: compute.PhaseAfterCull, Module: ir.CullKernel(), Kernel: "main",
		Buffers:      func(n string) (gpu.Buffer, bool) { b, ok := bufs[n]; return b, ok },
		IndirectArgs: func() (gpu.Buffer, int, bool) { return nil, 0, true }, // misconfigured
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	enc := dev.CreateCommandEncoder()
	err = pass.Record(compute.PassContext{Device: dev, Encoder: enc, Publish: func(compute.GPUResource) {}})
	if err == nil || !strings.Contains(err.Error(), "nil buffer") {
		t.Fatalf("expected nil-buffer error, got: %v", err)
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
