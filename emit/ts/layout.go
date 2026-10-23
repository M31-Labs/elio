package ts

import (
	"fmt"

	"m31labs.dev/elio/ir"
)

// FieldLayout is one uniform struct member with its byte offset and size.
type FieldLayout struct {
	Name   string
	Type   string
	Offset int
	Size   int
}

// StructLayout is a struct's fields and its total size in bytes.
type StructLayout struct {
	Struct string
	Fields []FieldLayout
	Size   int
}

func roundUp(n, a int) int { return (n + a - 1) / a * a }

// typeSizeAlign returns size and alignment under WGSL rules (std140 == true
// applies the std140 array/struct rounding, which only matters for the
// scalar-and-vector members uniform blocks hold in gather kernels).
func typeSizeAlign(m *ir.Module, t ir.Type, std140 bool) (size, align int, err error) {
	switch x := t.(type) {
	case ir.Scalar:
		if x.Name == "bool" {
			return 0, 0, fmt.Errorf("bool is not host-shareable")
		}
		return 4, 4, nil
	case ir.Vec:
		switch x.N {
		case 2:
			return 8, 8, nil
		case 3:
			return 12, 16, nil
		case 4:
			return 16, 16, nil
		}
	case ir.Mat:
		cs, ca, _ := typeSizeAlign(m, ir.Vec{N: x.Rows, Elem: x.Elem}, std140)
		stride := roundUp(cs, ca)
		if std140 {
			stride = roundUp(stride, 16)
			ca = 16
		}
		return stride * x.Cols, ca, nil
	case ir.Array:
		if x.Len == 0 {
			return 0, 0, fmt.Errorf("runtime-sized array in a uniform struct")
		}
		es, ea, err := typeSizeAlign(m, x.Elem, std140)
		if err != nil {
			return 0, 0, err
		}
		if std140 {
			ea = roundUp(ea, 16)
		}
		return roundUp(es, ea) * x.Len, ea, nil
	case ir.Named:
		l, err := layoutOf(m, x.Name, std140)
		if err != nil {
			return 0, 0, err
		}
		a := 4
		for _, s := range m.Structs {
			if s.Name == x.Name {
				for _, f := range s.Fields {
					_, fa, _ := typeSizeAlign(m, f.Type, std140)
					a = max(a, fa)
				}
			}
		}
		if std140 {
			a = roundUp(a, 16)
		}
		return l.Size, a, nil
	}
	return 0, 0, fmt.Errorf("type %T has no uniform layout", t)
}

func layoutOf(m *ir.Module, name string, std140 bool) (StructLayout, error) {
	for _, s := range m.Structs {
		if s.Name != name {
			continue
		}
		l := StructLayout{Struct: name}
		off, structAlign := 0, 4
		for _, f := range s.Fields {
			sz, al, err := typeSizeAlign(m, f.Type, std140)
			if err != nil {
				return StructLayout{}, fmt.Errorf("struct %s field %s: %w", name, f.Name, err)
			}
			off = roundUp(off, al)
			l.Fields = append(l.Fields, FieldLayout{Name: f.Name, Type: typeString(f.Type), Offset: off, Size: sz})
			off += sz
			structAlign = max(structAlign, al)
		}
		if std140 {
			structAlign = roundUp(structAlign, 16)
		}
		l.Size = roundUp(off, structAlign)
		return l, nil
	}
	return StructLayout{}, fmt.Errorf("struct %q not declared", name)
}

// wgslLayout computes the WGSL host-shareable layout of a struct.
func wgslLayout(m *ir.Module, name string) (StructLayout, error) { return layoutOf(m, name, false) }

// std140Layout computes the std140 layout the GLSL ES uniform block uses.
func std140Layout(m *ir.Module, name string) (StructLayout, error) { return layoutOf(m, name, true) }

// typeString spells a type in WGSL syntax.
func typeString(t ir.Type) string {
	switch x := t.(type) {
	case ir.Scalar:
		return x.Name
	case ir.Vec:
		return fmt.Sprintf("vec%d<%s>", x.N, x.Elem.Name)
	case ir.Mat:
		return fmt.Sprintf("mat%dx%d<%s>", x.Cols, x.Rows, x.Elem.Name)
	case ir.Named:
		return x.Name
	case ir.Array:
		if x.Len == 0 {
			return fmt.Sprintf("array<%s>", typeString(x.Elem))
		}
		return fmt.Sprintf("array<%s, %d>", typeString(x.Elem), x.Len)
	case ir.Atomic:
		return fmt.Sprintf("atomic<%s>", x.Elem.Name)
	case ir.StorageTexture:
		return "texture_storage_2d<" + x.Format + ", write>"
	case ir.Texture2D:
		return "texture_2d<f32>"
	}
	return "unknown"
}
