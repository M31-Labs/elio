package gosxtriad_test

import (
	"context"
	"strings"
	"testing"

	eliogosx "m31labs.dev/elio/adapter/gosx"
	elioir "m31labs.dev/elio/ir"
	eosartifact "m31labs.dev/eos/artifact/eos"
	eoscompiler "m31labs.dev/eos/compiler"
	eosruntime "m31labs.dev/eos/runtime"
	eoswebgpu "m31labs.dev/eos/runtime/backends/webgpu"
	"m31labs.dev/gosx/render/compute"
	"m31labs.dev/gosx/render/gpu"
	"m31labs.dev/gosx/render/gpu/headless"
	selenagosx "m31labs.dev/selena/adapter/gosx"
	selenahir "m31labs.dev/selena/hir"
)

func TestGoSXGPUTriadSameFrameContract(t *testing.T) {
	dev, _ := headless.New(64, 64)

	material, layout, err := selenagosx.Material(selenahir.DirectionalDiffuse(), nil)
	if err != nil {
		t.Fatalf("selena material: %v", err)
	}
	if material.CustomVertexWGSL == "" || material.CustomFragment == "" || layout.UniformBlock.Size == 0 {
		t.Fatalf("selena material did not populate GoSX shader slots/layout: material=%+v layout=%+v", material, layout)
	}

	elioPass := newElioCullPass(t, dev)
	if !strings.Contains(elioPass.WGSL(), "fn main(") {
		t.Fatalf("elio pass WGSL missing compute entry:\n%s", elioPass.WGSL())
	}

	eosBackend := eoswebgpu.New()
	eosBackend.SetExternalDevice(dev)
	eosBundle, err := eoscompiler.Build(nil, eoscompiler.Options{
		ModuleName: "triad_candidates",
		Preset:     eoscompiler.PresetTinyCandidates,
	})
	if err != nil {
		t.Fatalf("eos build: %v", err)
	}
	if !eosBackend.CanLoad(eosBundle.Artifact) {
		t.Fatalf("eos WebGPU backend cannot load artifact with backends=%v", eosBundle.Artifact.Requirements.SupportedBackends)
	}
	program, err := eosruntime.New(eosBackend).Load(context.Background(), eosBundle.Artifact)
	if err != nil {
		t.Fatalf("eos load: %v", err)
	}
	if got := program.Backend(); got != eosartifact.BackendWebGPU {
		t.Fatalf("eos backend = %q, want %q", got, eosartifact.BackendWebGPU)
	}

	frame := struct {
		Material     string
		UniformBytes int
		ComputePass  string
		Inference    eosartifact.BackendKind
	}{
		Material:     material.Name,
		UniformBytes: layout.UniformBlock.Size,
		ComputePass:  elioPass.ID(),
		Inference:    program.Backend(),
	}
	if frame.Material == "" || frame.UniformBytes == 0 || frame.ComputePass != "triad.cull" || frame.Inference != eosartifact.BackendWebGPU {
		t.Fatalf("incomplete same-frame contract: %+v", frame)
	}
}

func newElioCullPass(t *testing.T, dev *headless.Device) *eliogosx.Pass {
	t.Helper()
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

	pass, err := eliogosx.New(eliogosx.Config{
		ID:      "triad.cull",
		Phase:   compute.PhaseAfterCull,
		Module:  elioir.CullKernel(),
		Kernel:  "main",
		Buffers: func(name string) (gpu.Buffer, bool) { b, ok := bufs[name]; return b, ok },
		Groups:  func() (int, int, int) { return 1, 1, 1 },
		Publish: []compute.GPUResource{
			{Name: "triad.instances", Buffer: out, Role: compute.RoleInstanceAttr, Element: compute.InstanceRecordLayout(), Count: 4, Access: compute.Read},
			{Name: "triad.drawArgs", Buffer: args, Role: compute.RoleIndirectArgs, Element: compute.IndirectArgsLayout(), Count: 1, Access: compute.Read},
		},
	})
	if err != nil {
		t.Fatalf("elio gosx pass: %v", err)
	}
	return pass
}
