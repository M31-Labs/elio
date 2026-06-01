// Package cull is the M0 proof of the Manta↔Elio↔Selena triad seam: a
// render-coupled compute pass that lives OUTSIDE the GoSX renderer yet plugs
// into its frame through the compute.ExternalComputePass hook, dispatches on
// the renderer's command encoder, and publishes its compacted-instance and
// indirect-args buffers onto the GPU Resource Bus for the draw to consume.
//
// The kernel here is hand-written WGSL (a frustum culler, mirroring the
// engine's built-in cull). In the full triad this WGSL is what Elio's compiler
// emits from one .elio source — but the integration contract it exercises (hook
// + bus + canonical InstanceRecord layout) is identical, so this proves the
// seam before any compiler exists.
package cull

import (
	"encoding/binary"
	"fmt"
	"math"

	"m31labs.dev/gosx/render/compute"
	"m31labs.dev/gosx/render/gpu"
)

// cullWGSL: one thread per instance, frustum-test the bounding sphere, append
// survivors to a compacted output buffer and atomically bump the indirect
// instanceCount. Output records are the canonical 80-byte InstanceRecord.
const cullWGSL = `
struct CullUniforms {
  planes      : array<vec4<f32>, 6>,
  vertexCount : u32,
  radius      : f32,
  _pad0       : vec2<f32>,
};
struct InstanceRecord {
  model    : mat4x4<f32>,
  pickData : vec4<u32>,
};
@group(0) @binding(0) var<uniform> cull            : CullUniforms;
@group(0) @binding(1) var<storage, read>       input    : array<InstanceRecord>;
@group(0) @binding(2) var<storage, read_write> output   : array<InstanceRecord>;
@group(0) @binding(3) var<storage, read_write> drawArgs : array<atomic<u32>, 4>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input)) { return; }
  let record = input[i];
  let center = record.model[3].xyz;
  var inside = true;
  for (var p : i32 = 0; p < 6; p = p + 1) {
    let plane = cull.planes[p];
    if (dot(plane.xyz, center) + plane.w < -cull.radius) { inside = false; break; }
  }
  if (inside) {
    let slot = atomicAdd(&drawArgs[1], 1u);
    output[slot] = record;
  }
}
`

const cullUniformSize = 112 // 6×vec4 planes (96) + u32 + f32 + 8 pad

// resources are the GPU buffers + bind group for one mesh's cull, grown lazily.
type resources struct {
	capacity    int
	cullUniform gpu.Buffer
	inputBuf    gpu.Buffer
	outputBuf   gpu.Buffer
	drawArgsBuf gpu.Buffer
	bindGroup   gpu.BindGroup
}

// Pass is an out-of-tree ExternalComputePass that culls one instanced mesh.
type Pass struct {
	id          string
	dev         gpu.Device
	pipeline    gpu.ComputePipeline
	bgLayout    gpu.BindGroupLayout
	res         *resources
	instances   []byte // host-packed InstanceRecords (80 B each)
	count       int
	planes      [6][4]float32
	vertexCount uint32
	radius      float32
}

// New builds the cull compute pipeline on dev. id names the pass and its bus
// outputs ("<id>.instances", "<id>.drawArgs").
func New(dev gpu.Device, id string) (*Pass, error) {
	shader, err := dev.CreateShaderModule(gpu.ShaderDesc{SourceWGSL: cullWGSL, Label: "elio.cull:" + id})
	if err != nil {
		return nil, fmt.Errorf("elio/cull.New: shader: %w", err)
	}
	pipeline, err := dev.CreateComputePipeline(gpu.ComputePipelineDesc{
		Module: shader, EntryPoint: "main", AutoLayout: true, Label: "elio.cull:" + id,
	})
	if err != nil {
		return nil, fmt.Errorf("elio/cull.New: pipeline: %w", err)
	}
	return &Pass{id: id, dev: dev, pipeline: pipeline, bgLayout: pipeline.GetBindGroupLayout(0)}, nil
}

// SetFrame supplies this frame's culling inputs. instances is count packed
// 80-byte InstanceRecords; planes are the six world-space frustum planes;
// vertexCount seeds the indirect draw; radius is the conservative bound.
func (p *Pass) SetFrame(instances []byte, count int, planes [6][4]float32, vertexCount uint32, radius float32) {
	p.instances, p.count, p.planes, p.vertexCount, p.radius = instances, count, planes, vertexCount, radius
}

func (p *Pass) ID() string               { return p.id }
func (p *Pass) Phase() compute.PassPhase { return compute.PhaseAfterCull }

// Record dispatches the cull and publishes the compacted instance buffer +
// indirect args onto the bus. All queue writes happen before the compute pass
// is opened, per the PassContext contract.
func (p *Pass) Record(ctx compute.PassContext) error {
	if p.count == 0 {
		return nil
	}
	if err := p.ensure(p.count); err != nil {
		return err
	}
	q := ctx.Device.Queue()
	q.WriteBuffer(p.res.inputBuf, 0, p.instances)
	q.WriteBuffer(p.res.cullUniform, 0, cullUniformBytes(p.planes, p.vertexCount, p.radius))
	q.WriteBuffer(p.res.drawArgsBuf, 0, drawArgsResetBytes(p.vertexCount))

	pass := ctx.Encoder.BeginComputePass()
	pass.SetPipeline(p.pipeline)
	pass.SetBindGroup(0, p.res.bindGroup)
	pass.DispatchWorkgroups((p.count+63)/64, 1, 1)
	pass.End()

	ctx.Publish(compute.GPUResource{
		Name: p.id + ".instances", Buffer: p.res.outputBuf,
		Role: compute.RoleInstanceAttr, Element: compute.InstanceRecordLayout(),
		Count: p.count, Access: compute.Read,
	})
	ctx.Publish(compute.GPUResource{
		Name: p.id + ".drawArgs", Buffer: p.res.drawArgsBuf,
		Role: compute.RoleIndirectArgs, Element: compute.IndirectArgsLayout(),
		Count: 1, Access: compute.Read,
	})
	return nil
}

// ensure (re)creates buffers + bind group sized for at least count instances.
func (p *Pass) ensure(count int) error {
	if p.res != nil && p.res.capacity >= count {
		return nil
	}
	newCap := count + count/4
	if newCap < 32 {
		newCap = 32
	}
	bytes := newCap * compute.InstanceRecordStride

	mk := func(label string, size int, usage gpu.BufferUsage) (gpu.Buffer, error) {
		return p.dev.CreateBuffer(gpu.BufferDesc{Size: size, Usage: usage, Label: "elio.cull." + label + ":" + p.id})
	}
	inputBuf, err := mk("input", bytes, gpu.BufferUsageStorage|gpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	outputBuf, err := mk("output", bytes, gpu.BufferUsageStorage|gpu.BufferUsageVertex|gpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	drawArgsBuf, err := mk("drawArgs", compute.IndirectArgsStride, gpu.BufferUsageStorage|gpu.BufferUsageIndirect|gpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	cullUniform, err := mk("uniform", cullUniformSize, gpu.BufferUsageUniform|gpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	bg, err := p.dev.CreateBindGroup(gpu.BindGroupDesc{
		Layout: p.bgLayout,
		Entries: []gpu.BindGroupEntry{
			{Binding: 0, Buffer: cullUniform, Size: cullUniformSize},
			{Binding: 1, Buffer: inputBuf, Size: bytes},
			{Binding: 2, Buffer: outputBuf, Size: bytes},
			{Binding: 3, Buffer: drawArgsBuf, Size: compute.IndirectArgsStride},
		},
		Label: "elio.cull.bg:" + p.id,
	})
	if err != nil {
		return err
	}
	p.res = &resources{capacity: newCap, cullUniform: cullUniform, inputBuf: inputBuf, outputBuf: outputBuf, drawArgsBuf: drawArgsBuf, bindGroup: bg}
	return nil
}

func cullUniformBytes(planes [6][4]float32, vertexCount uint32, radius float32) []byte {
	out := make([]byte, cullUniformSize)
	for i := 0; i < 6; i++ {
		for j := 0; j < 4; j++ {
			binary.LittleEndian.PutUint32(out[i*16+j*4:], math.Float32bits(planes[i][j]))
		}
	}
	binary.LittleEndian.PutUint32(out[96:], vertexCount)
	binary.LittleEndian.PutUint32(out[100:], math.Float32bits(radius))
	return out
}

func drawArgsResetBytes(vertexCount uint32) []byte {
	out := make([]byte, compute.IndirectArgsStride)
	binary.LittleEndian.PutUint32(out[0:], vertexCount) // instanceCount/firstVertex/firstInstance stay 0
	return out
}

// staticCheck pins the compile-time guarantee that *Pass satisfies the hook.
var _ compute.ExternalComputePass = (*Pass)(nil)
