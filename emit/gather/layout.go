package gather

// Layout tells the host how to bind a gather kernel's resources. Every buffer
// is a 2D float texture of width elio_width texels; element i is texel
// (i % elio_width, i / elio_width).
type Layout struct {
	Kernel   string
	Inputs   []Buffer  // read buffers in binding order
	Outputs  []Output  // written buffers in binding order; Location is the color attachment
	Uniforms []Uniform // uniform blocks in binding order
}

// Buffer is a read-only input. A buffer (Kind "buffer") is a sampler2D of width
// elio_width plus a length uniform. A sampled texture (Kind "texture", an
// ir.Texture2D binding) is a sampler2D the host binds as the texture itself:
// it is already 2D, so it has no length uniform and no elio_width linearization.
type Buffer struct {
	Name    string // Elio binding name
	Kind    string // "buffer" or "texture"
	Elem    string // "f32", "vec2" or "vec4": how many channels carry data; a texture is "vec4"
	Sampler string // sampler2D uniform name, e.g. "src_tex"; a texture's is its binding name
	Length  string // uint uniform holding the element count, e.g. "src_len"; empty for a texture
	Binding int    // the Elio @binding number, for hosts that keep the WebGPU slot
}

// Output is a written buffer or storage texture: one RGBA color attachment.
// A storage texture's pixel (i % W, i / W) is the fragment's own pixel, so W
// must equal the pass width (elio_width). The host creates the render target in
// Format; the per-pixel byte cost of all outputs is at most 32.
type Output struct {
	Name     string // Elio binding name
	Elem     string // "f32", "vec2" or "vec4"; unused channels are zero
	Format   string // texture format ("rgba16float", "rgba32float", "r32float"); empty for a buffer
	Location int    // layout(location = N) of the out variable
	Variable string // GLSL out variable name, e.g. "dst_out"
}

// Uniform is a std140 uniform block.
type Uniform struct {
	Name    string // instance name (the Elio binding name)
	Block   string // block name for getUniformBlockIndex, e.g. "Params_ubo"
	Struct  string // Elio struct that defines the fields
	Binding int    // the Elio @binding number
}
