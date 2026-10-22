package ts

import (
	"testing"

	"m31labs.dev/elio/ir"
)

func vec(n int, e ir.Scalar) ir.Type { return ir.Vec{N: n, Elem: e} }

func TestWGSLUniformLayout(t *testing.T) {
	tests := []struct {
		name   string
		fields []ir.Field
		want   []FieldLayout
		size   int
		std140 int
	}{
		{
			name:   "scalars",
			fields: []ir.Field{{Name: "a", Type: ir.U32}, {Name: "b", Type: ir.F32}, {Name: "c", Type: ir.I32}},
			want:   []FieldLayout{{"a", "u32", 0, 4}, {"b", "f32", 4, 4}, {"c", "i32", 8, 4}},
			size:   12, std140: 16,
		},
		{
			name:   "vec2 alignment",
			fields: []ir.Field{{Name: "a", Type: ir.F32}, {Name: "v", Type: vec(2, ir.F32)}, {Name: "b", Type: ir.F32}},
			want:   []FieldLayout{{"a", "f32", 0, 4}, {"v", "vec2<f32>", 8, 8}, {"b", "f32", 16, 4}},
			size:   24, std140: 32,
		},
		{
			name:   "vec4 alignment and trailing padding",
			fields: []ir.Field{{Name: "a", Type: ir.U32}, {Name: "v", Type: vec(4, ir.F32)}, {Name: "b", Type: ir.U32}},
			want:   []FieldLayout{{"a", "u32", 0, 4}, {"v", "vec4<f32>", 16, 16}, {"b", "u32", 32, 4}},
			size:   48, std140: 48,
		},
		{
			name:   "vec3 then scalar packs",
			fields: []ir.Field{{Name: "v", Type: vec(3, ir.F32)}, {Name: "s", Type: ir.F32}},
			want:   []FieldLayout{{"v", "vec3<f32>", 0, 12}, {"s", "f32", 12, 4}},
			size:   16, std140: 16,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &ir.Module{Structs: []ir.Struct{{Name: "S", Fields: tc.fields}}}
			l, err := wgslLayout(m, "S")
			if err != nil {
				t.Fatal(err)
			}
			if len(l.Fields) != len(tc.want) {
				t.Fatalf("fields = %v", l.Fields)
			}
			for i, w := range tc.want {
				if l.Fields[i] != w {
					t.Errorf("field %d = %+v, want %+v", i, l.Fields[i], w)
				}
			}
			if l.Size != tc.size {
				t.Errorf("wgsl size = %d, want %d", l.Size, tc.size)
			}
			s, err := std140Layout(m, "S")
			if err != nil {
				t.Fatal(err)
			}
			if s.Size != tc.std140 {
				t.Errorf("std140 size = %d, want %d", s.Size, tc.std140)
			}
		})
	}
}

func TestLayoutUnknownStruct(t *testing.T) {
	if _, err := wgslLayout(&ir.Module{}, "Nope"); err == nil {
		t.Fatal("want error")
	}
}
