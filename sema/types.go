package sema

import (
	"fmt"

	"m31labs.dev/elio/ir"
)

// typeOf infers the type of e where it can be determined structurally — through
// Name lookups, struct-field access, swizzles, and indexing over the concrete
// types Elio already tracks (struct / vec / mat / array). It deliberately
// returns nil ("unknown") for literals, calls, and arithmetic rather than
// guessing, so type-dependent checks only fire when the type is certain.
func (c *checker) typeOf(s *scope, e ir.Expr) ir.Type {
	switch e := e.(type) {
	case ir.Name:
		if si, ok := s.lookup(e.Name); ok {
			return si.typ
		}
	case ir.Member:
		return c.memberType(c.typeOf(s, e.E), e.Field)
	case ir.Index:
		return elementType(c.typeOf(s, e.E))
	case ir.Call:
		if n, elem, ok := ir.VecConstructor(e.Func); ok {
			return ir.Vec{N: n, Elem: elem}
		}
		if elem, ok := ir.ScalarCast(e.Func); ok {
			return elem
		}
	}
	return nil
}

// memberType resolves obj.field: a struct field type, or a vector swizzle
// result. Returns nil when obj's type is unknown.
func (c *checker) memberType(obj ir.Type, field string) ir.Type {
	switch o := obj.(type) {
	case ir.Named:
		if fields, ok := c.structs[o.Name]; ok {
			return fields[field] // nil if missing — caller reports
		}
	case ir.Vec:
		if n := len(field); swizzleValid(field, o.N) {
			if n == 1 {
				return o.Elem
			}
			return ir.Vec{N: n, Elem: o.Elem}
		}
	}
	return nil
}

// elementType is the result of indexing arrays (element), vectors (scalar), and
// matrices (a column vector).
func elementType(obj ir.Type) ir.Type {
	switch o := obj.(type) {
	case ir.Array:
		return o.Elem
	case ir.Vec:
		return o.Elem
	case ir.Mat:
		return ir.Vec{N: o.Rows, Elem: o.Elem}
	}
	return nil
}

// swizzleComponent maps a swizzle char to its component index, or -1.
func swizzleComponent(b byte) int {
	switch b {
	case 'x', 'r':
		return 0
	case 'y', 'g':
		return 1
	case 'z', 'b':
		return 2
	case 'w', 'a':
		return 3
	}
	return -1
}

// swizzleValid reports whether field is a 1–4 char swizzle whose every
// component is in range for an N-component vector.
func swizzleValid(field string, n int) bool {
	if len(field) < 1 || len(field) > 4 {
		return false
	}
	for i := 0; i < len(field); i++ {
		idx := swizzleComponent(field[i])
		if idx < 0 || idx >= n {
			return false
		}
	}
	return true
}

// checkMember validates obj.field when obj's type is known: struct fields must
// exist; vector swizzles must be valid and in range; scalars have no members.
func (c *checker) checkMember(s *scope, m ir.Member) {
	obj := c.typeOf(s, m.E)
	if obj == nil {
		return // unknown object type — can't check, stay conservative
	}
	switch o := obj.(type) {
	case ir.Named:
		fields, ok := c.structs[o.Name]
		if !ok {
			return
		}
		if _, exists := fields[m.Field]; !exists {
			c.errf("struct %q has no field %q", o.Name, m.Field)
		}
	case ir.Vec:
		if !swizzleValid(m.Field, o.N) {
			c.errf("invalid swizzle %q on %s", m.Field, typeName(o))
		}
	case ir.Scalar:
		c.errf("%s has no field %q (it is a scalar)", typeName(o), m.Field)
	case ir.Atomic:
		c.errf("cannot access field %q of an atomic; load it first", m.Field)
	}
}

// typeName renders a type for diagnostics (close to .elio surface syntax).
func typeName(t ir.Type) string {
	switch t := t.(type) {
	case ir.Scalar:
		return t.Name
	case ir.Vec:
		// Elio surface suffixes: f32→"" (vec3), u32→"u" (vec3u), i32→"i".
		suffix := ""
		switch t.Elem.Name {
		case "u32":
			suffix = "u"
		case "i32":
			suffix = "i"
		}
		return fmt.Sprintf("vec%d%s", t.N, suffix)
	case ir.Mat:
		return fmt.Sprintf("mat%d", t.Cols)
	case ir.Array:
		if t.Len == 0 {
			return "[]" + typeName(t.Elem)
		}
		return fmt.Sprintf("[%d]%s", t.Len, typeName(t.Elem))
	case ir.Atomic:
		return "atomic_" + t.Elem.Name
	case ir.Named:
		return t.Name
	}
	return "?"
}
