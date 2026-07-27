module m31labs.dev/elio

go 1.26

require (
	github.com/odvcencio/gotreesitter v0.47.0
	m31labs.dev/gosx v0.25.10
	m31labs.dev/prism v0.1.3
)

require (
	github.com/klauspost/cpuid/v2 v2.0.9 // indirect
	lukechampine.com/blake3 v1.4.1 // indirect
	m31labs.dev/mll v0.1.0 // indirect
	m31labs.dev/turboquant v0.2.0 // indirect
)

require (
	m31labs.dev/eos v0.1.4
	m31labs.dev/selena v0.4.0
)

replace m31labs.dev/gosx => ../gosx // feat/browser-gpu-cull: ComputeExecutor registry
