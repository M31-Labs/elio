package gosx

import (
	"encoding/binary"
	"math"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/run"
	"m31labs.dev/gosx/render/gpu"
	"m31labs.dev/gosx/render/gpu/headless"
)

// HeadlessExecutor returns a headless.ComputeExecutor that runs the named
// kernel from module against the bound GPU buffers using the Elio CPU
// interpreter (run.Run). This bridges the headless device's compute registry
// to Elio's scalar interpreter, completing the GPU-vs-CPU parity gate:
//
//	dev.RegisterComputeExecutor("elio.cull", eliogosx.HeadlessExecutor(elioir.CullKernel(), "main"))
//
// Buffer contract (matching ir.CullKernel()):
//
//	binding 0  — CullUniforms (uniform): 6×vec4f32 planes (96 B) + u32 vertexCount (4 B) +
//	             f32 radius (4 B) + vec2f _pad0 (8 B) = 112 B total.
//	binding 1  — input storage (read):   []InstanceRecord, 80 B stride.
//	binding 2  — output storage (rw):    []InstanceRecord, 80 B stride.
//	binding 3  — drawArgs storage (rw):  4×u32 = 16 B.
//
// InstanceRecord layout (80 B):
//
//	bytes 0..63  — column-major mat4<f32> (model matrix)
//	bytes 64..79 — vec4<u32> (pickData)
//
// The executor derives its instance count from the input buffer's
// LastWriteSize (bytes written by Queue.WriteBuffer). This matches the
// production frame ordering: instance data is written before dispatch.
func HeadlessExecutor(module *ir.Module, kernel string) headless.ComputeExecutor {
	return &elioHeadlessExecutor{module: module, kernel: kernel}
}

type elioHeadlessExecutor struct {
	module *ir.Module
	kernel string
}

const instanceRecordStride = 80 // mat4<f32> (64 B) + vec4<u32> (16 B)

func (e *elioHeadlessExecutor) Exec(bindGroups map[int]*headless.BindGroup, _, _, _ int) {
	bg := bindGroups[0]
	if bg == nil {
		return
	}
	desc := bg.Desc()

	// Extract the four named buffers from bind group entries by binding slot.
	var (
		cullBuf     *headless.Buffer
		inputBuf    *headless.Buffer
		outputBuf   *headless.Buffer
		drawArgsBuf *headless.Buffer
	)
	for _, entry := range desc.Entries {
		buf, ok := entry.Buffer.(*headless.Buffer)
		if !ok || buf == nil {
			continue
		}
		switch entry.Binding {
		case 0:
			cullBuf = buf
		case 1:
			inputBuf = buf
		case 2:
			outputBuf = buf
		case 3:
			drawArgsBuf = buf
		}
	}
	if cullBuf == nil || inputBuf == nil || outputBuf == nil || drawArgsBuf == nil {
		return
	}

	cullData := cullBuf.Data()
	inputData := inputBuf.Data()
	outputData := outputBuf.Data()
	drawData := drawArgsBuf.Data()

	if len(drawData) < 16 {
		return
	}

	// Derive instance count from bytes written. If nothing was written yet
	// (LastWriteSize == 0), fall back to the full buffer size.
	inputSize := inputBuf.LastWriteSize()
	if inputSize <= 0 {
		inputSize = len(inputData)
	}
	inputSize -= inputSize % instanceRecordStride
	count := inputSize / instanceRecordStride
	if count <= 0 {
		return
	}

	// --- Decode CullUniforms (binding 0) ---
	// Layout: 6×vec4f32 (96 B) + u32 vertexCount (4 B) + f32 radius (4 B)
	if len(cullData) < 104 {
		return
	}
	planes := make([]any, 6)
	for p := 0; p < 6; p++ {
		off := p * 16
		nx := float64(math.Float32frombits(binary.LittleEndian.Uint32(cullData[off+0:])))
		ny := float64(math.Float32frombits(binary.LittleEndian.Uint32(cullData[off+4:])))
		nz := float64(math.Float32frombits(binary.LittleEndian.Uint32(cullData[off+8:])))
		w := float64(math.Float32frombits(binary.LittleEndian.Uint32(cullData[off+12:])))
		planes[p] = []float64{nx, ny, nz, w}
	}
	vertexCount := float64(binary.LittleEndian.Uint32(cullData[96:100]))
	radius := float64(math.Float32frombits(binary.LittleEndian.Uint32(cullData[100:104])))

	// Preserve the original drawArgs[0] (vertexCount from the draw call).
	daVertexCount := float64(binary.LittleEndian.Uint32(drawData[0:4]))
	if vertexCount == 0 {
		vertexCount = daVertexCount
	}

	// --- Decode input InstanceRecords (binding 1) ---
	in := make([]any, count)
	for i := 0; i < count; i++ {
		off := i * instanceRecordStride
		if off+instanceRecordStride > len(inputData) {
			break
		}
		mat := run.Mat{Cols: 4, Rows: 4, E: make([]float64, 16)}
		for j := 0; j < 16; j++ {
			mat.E[j] = float64(math.Float32frombits(binary.LittleEndian.Uint32(inputData[off+j*4:])))
		}
		pickData := []float64{
			float64(binary.LittleEndian.Uint32(inputData[off+64:])),
			float64(binary.LittleEndian.Uint32(inputData[off+68:])),
			float64(binary.LittleEndian.Uint32(inputData[off+72:])),
			float64(binary.LittleEndian.Uint32(inputData[off+76:])),
		}
		in[i] = map[string]any{"model": mat, "pickData": pickData}
	}

	// --- Execute via Elio CPU interpreter ---
	out := make([]any, count)
	drawArgsSlice := []float64{vertexCount, 0, 0, 0}

	mem := &run.Memory{Vars: map[string]any{
		"cull": map[string]any{
			"planes":      planes,
			"vertexCount": vertexCount,
			"radius":      radius,
		},
		"input":    in,
		"output":   out,
		"drawArgs": drawArgsSlice,
	}}

	if err := run.Run(e.module, e.kernel, count, mem); err != nil {
		// Execution failure: leave buffers untouched (safe no-op in tests).
		return
	}

	survivors := int(drawArgsSlice[1])

	// --- Write surviving InstanceRecords back to output buffer (binding 2) ---
	for i := 0; i < survivors && i < count; i++ {
		rec, ok := out[i].(map[string]any)
		if !ok {
			continue
		}
		mat, ok := rec["model"].(run.Mat)
		if !ok || len(mat.E) < 16 {
			continue
		}
		off := i * instanceRecordStride
		if off+instanceRecordStride > len(outputData) {
			break
		}
		for j := 0; j < 16; j++ {
			binary.LittleEndian.PutUint32(outputData[off+j*4:], math.Float32bits(float32(mat.E[j])))
		}
		if pd, ok := rec["pickData"].([]float64); ok && len(pd) >= 4 {
			for j := 0; j < 4; j++ {
				binary.LittleEndian.PutUint32(outputData[off+64+j*4:], uint32(pd[j]))
			}
		}
	}

	// --- Write drawArgs back to GPU buffer (binding 3) ---
	// drawArgs[0]: vertexCount (preserved from initial write)
	// drawArgs[1]: instanceCount (survivors from the kernel)
	binary.LittleEndian.PutUint32(drawData[0:4], uint32(daVertexCount))
	binary.LittleEndian.PutUint32(drawData[4:8], uint32(survivors))
}

// Ensure elioHeadlessExecutor satisfies headless.ComputeExecutor at compile time.
var _ headless.ComputeExecutor = (*elioHeadlessExecutor)(nil)

// Ensure gpu.Buffer can be type-asserted to *headless.Buffer at compile time.
// This is a static check that the executor's type assertion is valid.
var _ gpu.Buffer = (*headless.Buffer)(nil)
