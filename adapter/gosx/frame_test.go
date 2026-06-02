package gosx_test

import (
	"testing"

	elioir "m31labs.dev/elio/ir"
	eliogosx "m31labs.dev/elio/adapter/gosx"

	"m31labs.dev/gosx/engine"
	"m31labs.dev/gosx/render/bundle"
	"m31labs.dev/gosx/render/compute"
	"m31labs.dev/gosx/render/gpu"
	"m31labs.dev/gosx/render/gpu/headless"
)

// counting wraps an ExternalComputePass to record how many times the renderer
// invoked it during a frame.
type counting struct {
	inner compute.ExternalComputePass
	runs  int
}

func (c *counting) ID() string               { return c.inner.ID() }
func (c *counting) Phase() compute.PassPhase  { return c.inner.Phase() }
func (c *counting) Record(ctx compute.PassContext) error {
	c.runs++
	return c.inner.Record(ctx)
}

// TestElioCullPassDrivesFrame is the visible end-to-end: a real Elio cull kernel,
// wrapped as an ExternalComputePass and registered on the real bundle.Renderer,
// runs inside Frame() and publishes its compacted-instance + draw-args buffers
// under the mesh's bus key — so the renderer's draw resolves them via
// instanceDrawSource instead of the built-in cull. It proves an Elio kernel is a
// drop-in render-coupled compute stage for an actual GoSX scene.
func TestElioCullPassDrivesFrame(t *testing.T) {
	dev, surface := headless.New(400, 300)

	im := engine.RenderInstancedMesh{
		ID: "hero", Kind: "cube", VertexCount: 36, InstanceCount: 1,
		Transforms: []float64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}
	key := bundle.InstancedMeshKey(0, im)

	// Buffers for the cull kernel's four bindings, sized for the canonical
	// InstanceRecord layout. Seed the input so a CPU-backed cull has data.
	mkBuf := func(size int, usage gpu.BufferUsage) gpu.Buffer {
		b, err := dev.CreateBuffer(gpu.BufferDesc{Size: size, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		return b
	}
	const n = 1
	cull := mkBuf(256, gpu.BufferUsageUniform|gpu.BufferUsageCopyDst)
	input := mkBuf(compute.InstanceRecordStride*n, gpu.BufferUsageStorage|gpu.BufferUsageCopyDst)
	output := mkBuf(compute.InstanceRecordStride*n, gpu.BufferUsageStorage|gpu.BufferUsageVertex|gpu.BufferUsageCopyDst)
	drawArgs := mkBuf(compute.IndirectArgsStride, gpu.BufferUsageStorage|gpu.BufferUsageIndirect|gpu.BufferUsageCopyDst)
	bufs := map[string]gpu.Buffer{"cull": cull, "input": input, "output": output, "drawArgs": drawArgs}

	pass, err := eliogosx.New(eliogosx.Config{
		ID:      "elio.cull",
		Phase:   compute.PhaseAfterCull,
		Module:  elioir.CullKernel(),
		Kernel:  "main",
		Buffers: func(name string) (gpu.Buffer, bool) { b, ok := bufs[name]; return b, ok },
		Groups:  func() (int, int, int) { return 1, 1, 1 },
		Publish: []compute.GPUResource{
			{Name: key + ".instances", Buffer: output, Role: compute.RoleInstanceAttr, Element: compute.InstanceRecordLayout(), Count: n, Access: compute.Read},
			{Name: key + ".drawArgs", Buffer: drawArgs, Role: compute.RoleIndirectArgs, Element: compute.IndirectArgsLayout(), Count: 1, Access: compute.Read},
		},
	})
	if err != nil {
		t.Fatalf("adapter New: %v", err)
	}
	hook := &counting{inner: pass}

	r, err := bundle.New(bundle.Config{
		Device:                dev,
		Surface:               surface,
		ExternalComputePasses: []compute.ExternalComputePass{hook},
	})
	if err != nil {
		t.Fatalf("bundle.New: %v", err)
	}
	defer r.Destroy()

	b := engine.RenderBundle{
		Camera:          engine.RenderCamera{Z: 5, FOV: 1, Near: 0.1, Far: 100},
		InstancedMeshes: []engine.RenderInstancedMesh{im},
	}
	if err := r.Frame(b, 400, 300, 0); err != nil {
		t.Fatalf("Frame: %v", err)
	}

	if hook.runs != 1 {
		t.Fatalf("Elio cull pass ran %d times during the frame, want 1", hook.runs)
	}
}
