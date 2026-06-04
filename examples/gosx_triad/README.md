# GoSX GPU Triad Contract

This example assembles the three GPU-triad compilers around one GoSX frame owner:

- Selena lowers a material into GoSX custom shader slots.
- Elio lowers a compute kernel into a GoSX external compute pass.
- Eos loads a WebGPU artifact through the device-injection seam so inference can share the renderer-owned device.

The test is host-side and uses GoSX's headless GPU device. It proves the wiring and resource contracts without requiring a browser WebGPU device in CI. In a browser build, the same seam is `jsgpu.Device.NativeDevice()` passed to `eos/runtime/backends/webgpu.Backend.SetExternalDevice`.

Elio and Selena are still local workspace modules, so this example currently relies on the repository-level local replaces. Once their vanity imports are live, the example can move to a public GoSX module without sibling checkouts.
