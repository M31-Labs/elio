// Package graph describes a GPU pass graph: the resources, passes, bindings,
// uniform values and exports an effect needs, per backend and quality tier. A
// generic runner executes any valid graph, so an effect needs no host code of
// its own, only named parameter values.
//
// The types are JSON-compatible; `elio emit ts` writes them into the generated
// module. Build a graph with Builder, which validates it (Validate) before
// returning it. The contract is documented in the project notes as "Elio pass
// graph contract (v1)".
package graph

import (
	"encoding/json"

	"m31labs.dev/elio/ir"
)

// Version is the graph format version.
const Version = 1

// Backends and tiers.
const (
	WebGPU = "webgpu"
	WebGL2 = "webgl2"
)

// Param types. A buffer param has a type of the form "vec4<f32>[]" (an element
// type followed by "[]") and a Count.
const (
	F32  = "f32"
	U32  = "u32"
	I32  = "i32"
	Bool = "bool"
)

// Source kinds of a param.
const (
	SourceValue      = "value"      // set from the app; Value is the default
	SourceFrame      = "frame"      // supplied by the runner: time, dt, cameraX, cameraY, cameraZ
	SourceExpr       = "expr"       // Op over earlier params or numbers
	SourceAccumulate = "accumulate" // integrates the param Of by dt each frame
	SourceDelta      = "delta"      // the change of the param Of since the last frame (0 on the first)
)

// Graph is one pass graph for one backend and tier.
type Graph struct {
	Name       string            `json:"name"`
	Backend    string            `json:"backend"`
	Tier       string            `json:"tier"`
	Version    int               `json:"version"`
	Params     []Param           `json:"params"`
	Resources  []Resource        `json:"resources"`
	Passes     []Pass            `json:"passes"`
	FrameSwaps [][2]string       `json:"frameSwaps"`
	Exports    map[string]string `json:"exports"`
	// ExportParams lists numeric params the surface binds as uniforms (for
	// example a water offset wrapped per cascade).
	ExportParams []string `json:"exportParams,omitempty"`
}

// Param is a named per-frame input of the graph. Params are evaluated in
// declaration order, and a derived param refers only to earlier ones.
type Param struct {
	Name   string  `json:"name"`
	Type   string  `json:"type"`
	Count  int     `json:"count,omitempty"` // buffer params: element capacity
	Source Source  `json:"source"`
	Tween  float64 `json:"tween,omitempty"` // value params: seconds to move to a changed value
}

// Source says where a param's value comes from. Only the fields of its Kind
// are written to JSON.
type Source struct {
	Kind  string  `json:"kind"`
	Value float64 `json:"value"`          // value
	Name  string  `json:"name,omitempty"` // frame
	Op    string  `json:"op,omitempty"`   // expr: add, sub, mul, div, mod, min, max (two args), floor (one); mod is floored (the result takes the divisor's sign); the runner evaluates in f64
	Args  []any   `json:"args,omitempty"` // expr: "param:<name>" or a number
	Of    string  `json:"of,omitempty"`   // accumulate, delta: "param:<name>"
}

// MarshalJSON writes only the fields that belong to the source's kind.
func (s Source) MarshalJSON() ([]byte, error) {
	m := map[string]any{"kind": s.Kind}
	switch s.Kind {
	case SourceValue:
		m["value"] = s.Value
	case SourceFrame:
		m["name"] = s.Name
	case SourceExpr:
		m["op"], m["args"] = s.Op, s.Args
	case SourceAccumulate, SourceDelta:
		m["of"] = s.Of
	}
	return json.Marshal(m)
}

// Resource is a buffer or a texture. On WebGL2 a buffer is an RGBA32F texture
// Width texels wide and Count/Width tall; on WebGPU it is a storage buffer of
// Count elements.
type Resource struct {
	Name        string `json:"name"`
	Kind        string `json:"kind"` // "buffer" or "texture"
	ElementType string `json:"elementType,omitempty"`
	Count       int    `json:"count,omitempty"`
	Format      string `json:"format,omitempty"`
	Width       int    `json:"width,omitempty"`
	Height      int    `json:"height,omitempty"`
	MipLevels   int    `json:"mipLevels,omitempty"`
	// Feed names a buffer param whose values the runner uploads into this
	// buffer every frame.
	Feed string `json:"feed,omitempty"`
}

// Pass is a kernel pass, a mipmap pass or a repeat of other passes.
//
// A repeat runs Body Count times with the param named Index set to 0 to
// Count-1 (body uniform values refer to it as "param:<Index>"), and after
// each iteration swaps every pair in Swap (the names then refer to each
// other's resources, as with Graph.FrameSwaps). A runner may expand a repeat
// itself: Expand is the reference expansion, and Validate checks the
// expansion. Swaps must cancel by the end of the pass list, so a frame ends
// with every resource where it began (FrameSwaps aside).
type Pass struct {
	Name   string `json:"name"`
	Type   string `json:"type"` // "kernel", "mipmap" or "repeat"
	Kernel string `json:"kernel,omitempty"`
	Entry  string `json:"entry,omitempty"`
	// When is "always" (or empty, which is omitted from JSON), "param:<boolParam>" or "changed:<p1>,<p2>,...": the
	// pass runs on the first frame and on any frame when one of those params
	// changed (a tween counts).
	When     string    `json:"when,omitempty"`
	Dispatch *[3]int   `json:"dispatch,omitempty"` // WebGPU workgroup counts
	Target   *Target   `json:"target,omitempty"`   // WebGL2 render size
	Bindings []Binding `json:"bindings,omitempty"`
	Uniforms []Uniform `json:"uniforms,omitempty"`
	Resource string    `json:"resource,omitempty"` // mipmap: the texture

	Count int         `json:"count,omitempty"` // repeat: iterations
	Index string      `json:"index,omitempty"` // repeat: the iteration param
	Swap  [][2]string `json:"swap,omitempty"`  // repeat: resource pairs swapped after each iteration
	Body  []Pass      `json:"body,omitempty"`  // repeat: the passes to run
}

// Target is a WebGL2 pass's render size and the elio_width uniform.
type Target struct {
	Width     int `json:"width"`
	Height    int `json:"height"`
	ElioWidth int `json:"elioWidth"`
}

// Binding binds a kernel binding to a resource; Level picks the mip level of
// a texture. Inside a repeat, LevelAt replaces Level with an expression of the
// repeat's index.
type Binding struct {
	Name     string     `json:"name"`
	Resource string     `json:"resource"`
	Level    int        `json:"level,omitempty"` // 0 when omitted
	LevelAt  *IndexExpr `json:"levelAt,omitempty"`
}

// IndexExpr is a number computed from a repeat's index i, for a uniform value
// or a binding level inside the repeat's body: i + Offset, or, when Shr is
// set, max(Shr >> (i + Offset), 1) (the size of a mip level). A runner that
// expands repeats evaluates these with integer arithmetic.
type IndexExpr struct {
	Index  string `json:"index"`
	Offset int    `json:"offset,omitempty"`
	Shr    int    `json:"shr,omitempty"`
}

func (x IndexExpr) eval(i int) int {
	i += x.Offset
	if x.Shr != 0 {
		return max(x.Shr>>i, 1)
	}
	return i
}

// Uniform gives every field of a kernel's uniform struct a number or a
// "param:<name>" reference, or, inside a repeat, an IndexExpr. Fields whose names start with "_pad" may be
// omitted; the runner zero-fills them.
type Uniform struct {
	Name   string         `json:"name"`
	Values map[string]any `json:"values"`
}

// Kernel is a kernel the graph may use, as the kernel set provides it.
type Kernel struct {
	Name   string
	Entry  string // primary entry; for a gather kernel, the gather entry
	Module *ir.Module
	Gather bool
}
