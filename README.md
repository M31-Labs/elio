# Elio

**Render-coupled compute for the GoSX engine.** Elio is the third tier of the
GPU language triad — **Selena** shades (presentation / raster), **Elio**
computes per-frame (simulation, culling, geometry, screen-space, procedural:
buffers → buffers that feed the draw), **Manta** infers (neural graphics).

Named for the sun (Helios) to Selena's moon (Selene): Selena makes the light
behave; Elio drives the per-frame work that feeds it.

## Why

GoSX's renderer had a scene graph and a presentation pipeline but no portable
render-coupled compute: the particle integrator was hand-written in four places
(WebGPU-Go, WebGL-JS, Android-Kotlin, iOS-Swift), mobile had no GPU compute,
and there was no GPU-driven rendering, portable post-FX, or simulation. Elio is
that tier — author a compute pass once, emit WGSL / MSL / SPIR-V (+ a CPU
fallback), and let the GoSX frame schedule it.

## How it plugs in

Elio targets GoSX's `render/gpu` package as the single device of record and
plugs into the renderer through `render/compute.ExternalComputePass`: a pass
records its dispatch onto the frame's command encoder and **publishes** its
output buffers onto the **GPU Resource Bus** (`render/compute.GPUResource`),
which the draw consumes. The canonical `InstanceRecord` layout (80 B = mat4 +
vec4&lt;u32&gt;) maps 1:1 to the engine's existing instanced-draw path, so an Elio
kernel's output binds with no renderer change.

## Status — M0

`examples/cull` is a complete, building out-of-tree `ExternalComputePass` (a
frustum culler) that dispatches on the renderer's encoder and publishes its
compacted-instance + indirect-args buffers onto the bus. The kernel WGSL is
hand-written today; the Elio compiler (`.elio` → WGSL/MSL) is what will generate
it. This example proves the integration seam before the compiler exists.

## Layout

| Path | Role |
|---|---|
| `examples/cull/` | M0 proof: an out-of-tree render-coupled compute pass driving the GoSX frame |

Knowledge space: `hypha://m31labs/elio`.
