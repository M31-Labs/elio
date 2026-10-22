package graph

import (
	"errors"
	"fmt"
	"strings"

	"m31labs.dev/elio/ir"
)

// Builder assembles a Graph for one backend and tier. Methods chain and keep
// the first error; Build returns it, or the validation errors of the finished
// graph. Params must be declared before anything that refers to them, and a
// derived param after the params it uses.
//
//	b := graph.NewBuilder("water", graph.WebGPU, "high", kernels)
//	b.Value("speed", graph.F32, 2)                       // set from the app
//	b.Frame("time", "time")                              // supplied by the runner
//	b.Expr("phase", graph.F32, "mod", graph.Ref("time"), 60.0)
//	b.Texture("height", "rgba16float", 256, 256, 1)
//	b.Pass(graph.Pass{Name: "step", Type: "kernel", Kernel: "water_step", Entry: "water_step",
//	    When: "always", Dispatch: b.Workgroups("water_step", "water_step", 256*256),
//	    Bindings: []graph.Binding{{Name: "out", Resource: "height"}},
//	    Uniforms: []graph.Uniform{b.Uniform("water_step", "p", map[string]any{"t": graph.Ref("phase")})}})
//	b.Export("height", "height")
//	g, err := b.Build()
type Builder struct {
	g       Graph
	kernels []Kernel
	err     error
}

// NewBuilder starts a graph. kernels are the kernels its passes may use.
func NewBuilder(name, backend, tier string, kernels []Kernel) *Builder {
	return &Builder{
		g: Graph{Name: name, Backend: backend, Tier: tier, Version: Version,
			Params: []Param{}, Resources: []Resource{}, Passes: []Pass{}, FrameSwaps: [][2]string{}, Exports: map[string]string{}},
		kernels: kernels,
	}
}

func (b *Builder) fail(format string, a ...any) {
	if b.err == nil {
		b.err = fmt.Errorf("graph %s: %s", b.g.Name, fmt.Sprintf(format, a...))
	}
}

// Ref returns the reference string for a param, for uniform values, expression
// arguments and pass conditions.
func Ref(param string) string { return "param:" + param }

// Changed returns the pass condition that runs a pass on the first frame and
// whenever one of the params changed.
func Changed(params ...string) string { return "changed:" + strings.Join(params, ",") }

// Value declares a param the app sets (through scene updates), with its
// default.
func (b *Builder) Value(name, typ string, def float64) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: typ, Source: Source{Kind: SourceValue, Value: def}})
	return b
}

// TweenedValue is Value, and a change moves the param to its new value over
// tween seconds of presentation time.
func (b *Builder) TweenedValue(name, typ string, def, tween float64) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: typ, Source: Source{Kind: SourceValue, Value: def}, Tween: tween})
	return b
}

// BufferParam declares an array of up to count elements the app supplies; a
// resource with Feed set to the name receives it every frame.
func (b *Builder) BufferParam(name, elem string, count int) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: elem + "[]", Count: count, Source: Source{Kind: SourceValue}})
	return b
}

// Frame declares an f32 param the runner supplies: time, dt, cameraX,
// cameraY or cameraZ.
func (b *Builder) Frame(name, source string) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: F32, Source: Source{Kind: SourceFrame, Name: source}})
	return b
}

// Expr declares a param computed each frame by op (add, sub, mul, div, mod,
// min, max over two args; floor over one) from earlier params (Ref) or
// numbers.
func (b *Builder) Expr(name, typ, op string, args ...any) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: typ, Source: Source{Kind: SourceExpr, Op: op, Args: args}})
	return b
}

// Accumulate declares an f32 param that integrates the earlier param of by dt
// every frame (of is the plain param name).
func (b *Builder) Accumulate(name, of string) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: F32, Source: Source{Kind: SourceAccumulate, Of: Ref(of)}})
	return b
}

// Delta declares a param holding the change of the earlier param of since the
// last frame, 0 on the first frame. Use it for "how far did the window move".
func (b *Builder) Delta(name, typ, of string) *Builder {
	b.g.Params = append(b.g.Params, Param{Name: name, Type: typ, Source: Source{Kind: SourceDelta, Of: Ref(of)}})
	return b
}

// Buffer declares a buffer resource. width is the texel width of a WebGL2
// buffer (ignored on WebGPU).
func (b *Builder) Buffer(name, elem string, count, width int) *Builder {
	r := Resource{Name: name, Kind: "buffer", ElementType: elem, Count: count}
	if b.g.Backend == WebGL2 {
		r.Width = width
	}
	b.g.Resources = append(b.g.Resources, r)
	return b
}

// FedBuffer declares a buffer the runner fills each frame from a buffer param.
func (b *Builder) FedBuffer(name, elem string, count, width int, param string) *Builder {
	b.Buffer(name, elem, count, width)
	b.g.Resources[len(b.g.Resources)-1].Feed = param
	return b
}

// Texture declares a texture resource with mips levels (MipCount for a full chain).
func (b *Builder) Texture(name, format string, width, height, mips int) *Builder {
	b.g.Resources = append(b.g.Resources, Resource{Name: name, Kind: "texture", Format: format, Width: width, Height: height, MipLevels: mips})
	return b
}

// MipCount is the length of a full mip chain.
func MipCount(width, height int) int {
	n := 1
	for s := max(width, height); s > 1; s >>= 1 {
		n++
	}
	return n
}

// Pass appends a pass.
func (b *Builder) Pass(p Pass) *Builder {
	b.g.Passes = append(b.g.Passes, p)
	return b
}

// Swap makes the runner swap two same-shaped resources after each frame.
func (b *Builder) Swap(a, c string) *Builder {
	b.g.FrameSwaps = append(b.g.FrameSwaps, [2]string{a, c})
	return b
}

// Export names a resource for the surface to bind.
func (b *Builder) Export(name, resource string) *Builder {
	b.g.Exports[name] = resource
	return b
}

func (b *Builder) kernel(name string) (Kernel, bool) {
	for _, k := range b.kernels {
		if k.Name == name {
			return k, true
		}
	}
	b.fail("unknown kernel %q", name)
	return Kernel{}, false
}

// Workgroups returns the WebGPU dispatch that gives a kernel entry at least
// elements invocations along x. Validate rejects more than 65,535 workgroups.
func (b *Builder) Workgroups(kernel, entry string, elements int) *[3]int {
	k, ok := b.kernel(kernel)
	if !ok {
		return nil
	}
	for _, e := range k.Module.Kernels {
		if e.Name == entry {
			per := e.WorkgroupSize[0]
			groups := (elements + per - 1) / per
			return &[3]int{groups, 1, 1}
		}
	}
	b.fail("kernel %q has no entry %q", kernel, entry)
	return nil
}

// Uniform returns the uniform entry for a kernel's uniform block. A field
// named "pad" that the caller omits gets 0; fields starting "_pad" stay
// omitted (the runner zero-fills them).
func (b *Builder) Uniform(kernel, binding string, values map[string]any) Uniform {
	u := Uniform{Name: binding, Values: map[string]any{}}
	for k, v := range values {
		u.Values[k] = v
	}
	k, ok := b.kernel(kernel)
	if !ok {
		return u
	}
	for _, mb := range k.Module.Bindings {
		n, isNamed := mb.Type.(ir.Named)
		if mb.Name != binding || !isNamed {
			continue
		}
		for _, s := range k.Module.Structs {
			if s.Name != n.Name {
				continue
			}
			for _, f := range s.Fields {
				if _, set := u.Values[f.Name]; !set && f.Name == "pad" {
					u.Values[f.Name] = 0
				}
			}
		}
	}
	return u
}

// Build validates and returns the graph.
func (b *Builder) Build() (Graph, error) {
	if b.err != nil {
		return Graph{}, b.err
	}
	if err := Validate(b.g, b.kernels); err != nil {
		return Graph{}, errors.Join(fmt.Errorf("graph %s/%s is invalid", b.g.Backend, b.g.Tier), err)
	}
	return b.g, nil
}
