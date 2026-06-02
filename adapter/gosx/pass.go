// Package gosx adapts an Elio compute kernel into a GoSX render-coupled compute
// pass. It compiles an ir.Module's kernel to WGSL and implements
// render/compute.ExternalComputePass, deriving each bind group directly from the
// kernel's own @group/@binding declarations — so an Elio kernel drops into the
// renderer's frame (at a chosen PassPhase) with no hand-written GPU plumbing,
// and its outputs are published onto the resource bus for the draw to consume.
//
// This is the render-coupled seam the triad is built around: Elio owns the
// compute kernel, GoSX owns the frame and the buffers, and the bus is the
// contract between them (see m31labs.dev/gosx/render/compute).
package gosx

import (
	"fmt"

	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/gosx/render/compute"
	"m31labs.dev/gosx/render/gpu"
)

// Config builds a Pass.
type Config struct {
	ID     string            // stable pass id (diagnostics, output naming)
	Phase  compute.PassPhase // where in the frame the pass runs
	Module *ir.Module        // the compiled unit
	Kernel string            // entry kernel name within Module

	// Buffers resolves an Elio binding name to the GPU buffer to bind. The caller
	// wires these from the renderer's resource bus (uniforms it owns, storage it
	// allocates). Required.
	Buffers func(binding string) (gpu.Buffer, bool)
	// Groups returns the workgroup dispatch counts for the frame. Defaults to
	// (1,1,1) when nil.
	Groups func() (x, y, z int)
	// Publish lists resources to announce on the bus after the dispatch, so later
	// passes and the draw can resolve them by name.
	Publish []compute.GPUResource
}

// Pass is an Elio kernel adapted as a render/compute.ExternalComputePass.
type Pass struct {
	cfg      Config
	kern     *ir.Kernel
	wgsl     string
	pipeline gpu.ComputePipeline // built lazily on first Record (needs a Device)
}

// New compiles cfg.Kernel from cfg.Module to WGSL and returns the pass.
func New(cfg Config) (*Pass, error) {
	if cfg.Module == nil {
		return nil, fmt.Errorf("elio/gosx: nil module")
	}
	if cfg.Buffers == nil {
		return nil, fmt.Errorf("elio/gosx: Config.Buffers is required")
	}
	var kern *ir.Kernel
	for i := range cfg.Module.Kernels {
		if cfg.Module.Kernels[i].Name == cfg.Kernel {
			kern = &cfg.Module.Kernels[i]
			break
		}
	}
	if kern == nil {
		return nil, fmt.Errorf("elio/gosx: kernel %q not found in module", cfg.Kernel)
	}
	src, err := wgsl.Emit(cfg.Module)
	if err != nil {
		return nil, fmt.Errorf("elio/gosx: emit wgsl: %w", err)
	}
	return &Pass{cfg: cfg, kern: kern, wgsl: src}, nil
}

// ID reports the pass's stable identifier.
func (p *Pass) ID() string { return p.cfg.ID }

// Phase reports where in the frame the pass runs.
func (p *Pass) Phase() compute.PassPhase { return p.cfg.Phase }

// WGSL returns the compiled shader source (for validation / debugging).
func (p *Pass) WGSL() string { return p.wgsl }

// Record builds the pipeline (once), binds the kernel's resources, dispatches,
// and publishes the configured outputs onto the bus. It opens and closes its own
// compute pass on ctx.Encoder, per the ExternalComputePass contract.
func (p *Pass) Record(ctx compute.PassContext) error {
	if p.pipeline == nil {
		sh, err := ctx.Device.CreateShaderModule(gpu.ShaderDesc{SourceWGSL: p.wgsl, Label: p.cfg.ID})
		if err != nil {
			return fmt.Errorf("elio/gosx: %s: shader module: %w", p.cfg.ID, err)
		}
		pl, err := ctx.Device.CreateComputePipeline(gpu.ComputePipelineDesc{
			Module: sh, EntryPoint: p.kern.Name, AutoLayout: true, Label: p.cfg.ID,
		})
		if err != nil {
			return fmt.Errorf("elio/gosx: %s: compute pipeline: %w", p.cfg.ID, err)
		}
		p.pipeline = pl
	}

	// One bind group per @group, in first-seen order; each entry's binding slot
	// and the kernel's WGSL @binding match by construction.
	byGroup := map[int][]gpu.BindGroupEntry{}
	var order []int
	for _, b := range p.cfg.Module.Bindings {
		buf, ok := p.cfg.Buffers(b.Name)
		if !ok {
			return fmt.Errorf("elio/gosx: %s: no buffer bound for %q", p.cfg.ID, b.Name)
		}
		if _, seen := byGroup[b.Group]; !seen {
			order = append(order, b.Group)
		}
		byGroup[b.Group] = append(byGroup[b.Group], gpu.BindGroupEntry{Binding: b.Binding, Buffer: buf})
	}

	pass := ctx.Encoder.BeginComputePass()
	pass.SetPipeline(p.pipeline)
	for _, g := range order {
		bg, err := ctx.Device.CreateBindGroup(gpu.BindGroupDesc{
			Layout:  p.pipeline.GetBindGroupLayout(g),
			Entries: byGroup[g],
			Label:   p.cfg.ID,
		})
		if err != nil {
			pass.End()
			return fmt.Errorf("elio/gosx: %s: bind group %d: %w", p.cfg.ID, g, err)
		}
		pass.SetBindGroup(g, bg)
	}
	x, y, z := 1, 1, 1
	if p.cfg.Groups != nil {
		x, y, z = p.cfg.Groups()
	}
	pass.DispatchWorkgroups(x, y, z)
	pass.End()

	for _, r := range p.cfg.Publish {
		ctx.Publish(r)
	}
	return nil
}
