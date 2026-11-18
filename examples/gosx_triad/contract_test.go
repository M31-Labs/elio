package gosxtriad_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
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
	"m31labs.dev/selena"
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

	elioPass, buffers := newElioCullPass(t, dev)
	verifyCullToMaterial(t, dev, elioPass, buffers)
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

func newElioCullPass(t *testing.T, dev *headless.Device) (*eliogosx.Pass, map[string]gpu.Buffer) {
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
	mk("cull", 112, gpu.BufferUsageUniform|gpu.BufferUsageCopyDst)
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
	return pass, bufs
}

// Run the cull and verify the surviving record can feed a Selena vertex
// stage directly. Picking metadata follows the four matrix columns.
func verifyCullToMaterial(t *testing.T, dev *headless.Device, pass *eliogosx.Pass, buffers map[string]gpu.Buffer) {
	t.Helper()
	record := compute.InstanceRecordLayout()
	const source = `material CulledInstance {
    instance m0 : vec4
    instance m1 : vec4
    instance m2 : vec4
    instance m3 : vec4
    vertex(geo) -> vec4 {
        return mvp * (m0 * geo.position.x + m1 * geo.position.y + m2 * geo.position.z + m3)
    }
    surface(geo) -> color { return rgb(0.2, 0.6, 1.0) }
}`
	compiled, err := selena.Compile([]byte(source), selena.CompileOptions{FirstInstanceLocation: record.Fields[0].Location, Targets: []selena.Target{selena.TargetWGSL}})
	if err != nil {
		t.Fatal(err)
	}
	attrs := compiled.Layout.InstanceAttributes
	if len(attrs) != 4 {
		t.Fatalf("instance inputs = %+v", attrs)
	}
	for i, attr := range attrs {
		field := record.Fields[i]
		if attr.Name != field.Name || attr.Location != field.Location || attr.Offset != field.Offset || attr.Type != "vec4" || field.Format != gpu.VertexFormatFloat32x4 {
			t.Fatalf("Selena input %+v does not match Elio output %+v", attr, field)
		}
	}
	planes := make([]byte, 112)
	binary.LittleEndian.PutUint32(planes, math.Float32bits(1)) // keep x >= 0
	binary.LittleEndian.PutUint32(planes[96:], 3)
	input := make([]byte, 2*record.Stride)
	for i, x := range []float32{-2, 3} {
		instance := input[i*record.Stride:]
		for col := 0; col < 4; col++ {
			binary.LittleEndian.PutUint32(instance[col*20:], math.Float32bits(1))
		}
		binary.LittleEndian.PutUint32(instance[48:], math.Float32bits(x))
		binary.LittleEndian.PutUint32(instance[64:], uint32(40+i))
	}
	dev.Queue().WriteBuffer(buffers["cull"], 0, planes)
	dev.Queue().WriteBuffer(buffers["input"], 0, input)
	dev.Queue().WriteBuffer(buffers["drawArgs"], 0, []byte{3, 0, 0, 0})
	dev.RegisterComputeExecutor(pass.ID(), eliogosx.HeadlessExecutor(elioir.CullKernel(), "main"))
	published := map[string]compute.GPUResource{}
	encoder := dev.CreateCommandEncoder()
	if err := pass.Record(compute.PassContext{Device: dev, Encoder: encoder, Publish: func(r compute.GPUResource) { published[r.Name] = r }}); err != nil {
		t.Fatal(err)
	}
	dev.Queue().Submit(encoder.Finish())
	instances, args := published["triad.instances"], published["triad.drawArgs"]
	if instances.Buffer != buffers["output"] || args.Buffer != buffers["drawArgs"] || instances.Element.Stride != record.Stride {
		t.Fatalf("unexpected published resources: %+v", published)
	}
	draws := args.Buffer.(*headless.Buffer).Data()
	if binary.LittleEndian.Uint32(draws) != 3 || binary.LittleEndian.Uint32(draws[4:]) != 1 {
		t.Fatalf("expected 3 vertices, 1 visible instance; indirect args %v", draws)
	}
	if got := instances.Buffer.(*headless.Buffer).Data()[:record.Stride]; !bytes.Equal(got, input[record.Stride:]) {
		t.Fatalf("compacted record differs from the visible transform and pick ID: %x", got)
	}
}
