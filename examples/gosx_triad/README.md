# GoSX GPU Triad Contract

This example assembles the three GPU-triad compilers around one GoSX frame owner:

- Selena lowers a material into GoSX custom shader slots.
- Elio lowers a compute kernel into a GoSX external compute pass.
- Eos loads a WebGPU artifact through the device-injection seam so inference can share the renderer-owned device.

The test is host-side and uses GoSX's headless GPU device. It proves the wiring and resource contracts without requiring a browser WebGPU device in CI. In a browser build, the same seam is `jsgpu.Device.NativeDevice()` passed to `eos/runtime/backends/webgpu.Backend.SetExternalDevice`.

Run the example with `GOWORK=off go test ./examples/gosx_triad`. It uses the
published module versions in `go.mod`. The Elio cull rejects one instance and
publishes the other, including its pick ID. The Selena shader reads the four
matrix columns at locations 4 through 7. Use the producer's 80-byte stride
when binding that stream. The four columns occupy the first 64 bytes and
GoSX keeps 16 bytes of picking metadata after them. The headless test checks
the compacted bytes, indirect draw arguments and compiler binding layout.

When pooling instance buffers, bind exactly the live record range to the
compute kernel. Its WGSL guard reads `arrayLength(&input)`, which WebGPU
calculates from the binding's byte length. Binding all 32 slots for two
records can send 31 survivors to the indirect draw when only one passes the
cull. The pass in `examples/cull` keeps the allocation and updates the range
as the population changes.
