package stdlib

import (
	_ "embed"
	"fmt"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/parse"
)

// GPUDrivenCullSource is the .elio source of the Scene3D GPU-driven instance
// cull: frustum, optional distance and contribution culls, two-phase Hi-Z
// occlusion, and shadow-light views. GoSX embeds the WGSL that
// emit/wgsl produces from it in client/runtime/scene3d/indirect-instancing.ts.
//
//go:embed gpudriven/cull.elio
var GPUDrivenCullSource string

// GPUDrivenHiZDownsampleSource is the .elio source of one clamped Hi-Z pyramid
// level over a single packed buffer.
//
//go:embed gpudriven/hiz_downsample.elio
var GPUDrivenHiZDownsampleSource string

// GPUDrivenCull parses GPUDrivenCullSource. Entry kernel: "cull".
func GPUDrivenCull() (*ir.Module, error) {
	mod, err := parse.Parse(GPUDrivenCullSource)
	if err != nil {
		return nil, fmt.Errorf("stdlib: gpudriven/cull.elio: %w", err)
	}
	return mod, nil
}

// GPUDrivenHiZDownsample parses GPUDrivenHiZDownsampleSource. Entry kernel:
// "downsample".
func GPUDrivenHiZDownsample() (*ir.Module, error) {
	mod, err := parse.Parse(GPUDrivenHiZDownsampleSource)
	if err != nil {
		return nil, fmt.Errorf("stdlib: gpudriven/hiz_downsample.elio: %w", err)
	}
	return mod, nil
}
