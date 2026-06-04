package prismtypes

import (
	"m31labs.dev/elio/ir"
	"m31labs.dev/prism/gputype"
)

// FromIR converts the Elio IR types Prism understands into gputype.Type.
// Backend-local typeName functions still handle Atomic, Named, and arrays
// whose element is one of those backend-specific types.
func FromIR(t ir.Type) (gputype.Type, bool) {
	switch x := t.(type) {
	case ir.Scalar:
		return gputype.Scalar{Name: x.Name}, true
	case ir.Vec:
		return gputype.Vec{N: x.N, Elem: gputype.Scalar{Name: x.Elem.Name}}, true
	case ir.Mat:
		return gputype.Mat{Cols: x.Cols, Rows: x.Rows, Elem: gputype.Scalar{Name: x.Elem.Name}}, true
	case ir.Array:
		elem, ok := FromIR(x.Elem)
		if !ok {
			return nil, false
		}
		return gputype.Array{Elem: elem, Len: x.Len}, true
	}
	return nil, false
}
