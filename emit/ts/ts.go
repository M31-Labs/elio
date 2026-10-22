// Package ts emits a TypeScript module from a set of Elio kernels: each
// kernel's WGSL source, its WebGL2 gather GLSL ES source and layout (when the
// kernel is gather-compatible), its workgroup size, its binding table and the
// byte layout of its uniform structs. A browser host imports the one module
// instead of keeping hand-copied shader strings.
package ts

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"m31labs.dev/elio/emit/gather"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/sema"
)

// Kernel is one exported kernel. Module may hold several entry points (the
// FFT compute module does); Entry names the primary one, and may be empty when
// the module has exactly one kernel. The WGSL is the whole module.
type Kernel struct {
	Name   string // snake_case export stem, e.g. "ocean_mip" exports oceanMipWGSL
	Entry  string
	Module *ir.Module
	Gather bool // also emit the GLSL ES gather source and layout for Entry
}

// KernelSet is the group of kernels written to one module.
type KernelSet struct {
	Name    string
	Kernels []Kernel
}

// Meta is the header information of the module.
type Meta struct {
	Version string // Elio version string
	// Global, when set, selects the classic-script form: an IIFE that assigns
	// one deeply frozen object to window.<Global> instead of ES exports. A
	// bundler that concatenates scripts into an IIFE chunk cannot consume
	// `export`. Global must be a plain identifier.
	Global string
}

type bindingJSON struct {
	Group       int    `json:"group"`
	Binding     int    `json:"binding"`
	Name        string `json:"name"`
	Kind        string `json:"kind"`
	Format      string `json:"format,omitempty"`
	ElementType string `json:"elementType"`
}

type entryJSON struct {
	Entry     string `json:"entry"`
	Workgroup [3]int `json:"workgroup"`
}

type uniformJSON struct {
	Group   int         `json:"group"`
	Binding int         `json:"binding"`
	Name    string      `json:"name"`
	Struct  string      `json:"struct"`
	Size    int         `json:"size"`
	Fields  []fieldJSON `json:"fields"`
}

type fieldJSON struct {
	Name   string `json:"name"`
	Type   string `json:"type"`
	Offset int    `json:"offset"`
	Size   int    `json:"size"`
}

type gatherBufferJSON struct {
	Name    string `json:"name"`
	Kind    string `json:"kind"`
	Elem    string `json:"elem"`
	Sampler string `json:"sampler"`
	Length  string `json:"length"`
	Binding int    `json:"binding"`
}

type gatherOutputJSON struct {
	Name     string `json:"name"`
	Elem     string `json:"elem"`
	Format   string `json:"format"`
	Location int    `json:"location"`
	Variable string `json:"variable"`
}

type gatherUniformJSON struct {
	Name    string `json:"name"`
	Block   string `json:"block"`
	Struct  string `json:"struct"`
	Binding int    `json:"binding"`
}

type gatherLayoutJSON struct {
	Kernel   string              `json:"kernel"`
	Inputs   []gatherBufferJSON  `json:"inputs"`
	Outputs  []gatherOutputJSON  `json:"outputs"`
	Uniforms []gatherUniformJSON `json:"uniforms"`
}

// member is one emitted value of a kernel. ES mode writes it as
// `export const <stem><Suffix>[: Type] = Lit;`; global mode writes it as a
// property `<key>: Lit,` of the kernel's object.
type member struct {
	key, suffix, typ, lit string
}

var identRE = regexp.MustCompile(`^[A-Za-z_$][A-Za-z0-9_$]*$`)

// Emit renders set as one TypeScript module, or as one classic script when
// meta.Global is set. The output is deterministic.
func Emit(set KernelSet, meta Meta) (string, error) {
	if meta.Global != "" && !identRE.MatchString(meta.Global) {
		return "", fmt.Errorf("ts: global name %q is not an identifier", meta.Global)
	}
	var body strings.Builder
	h := sha256.New()
	seen := map[string]bool{}
	for _, k := range set.Kernels {
		stem := camel(k.Name)
		if seen[stem] {
			return "", fmt.Errorf("ts: duplicate kernel name %q", k.Name)
		}
		seen[stem] = true
		if k.Module == nil {
			return "", fmt.Errorf("ts: kernel %q has no module", k.Name)
		}
		if errs := sema.Check(k.Module); len(errs) > 0 {
			return "", fmt.Errorf("ts: kernel %q: %w", k.Name, errors.Join(errs...))
		}
		entry, err := primary(k)
		if err != nil {
			return "", err
		}
		src, err := wgsl.Emit(k.Module)
		if err != nil {
			return "", fmt.Errorf("ts: kernel %q: wgsl: %w", k.Name, err)
		}
		h.Write([]byte(src))
		ms := []member{
			{"wgsl", "WGSL", "", "`" + escapeTemplate(src) + "`"},
			{"entry", "Entry", "", quote(entry.Name)},
			{"workgroup", "Workgroup", "", fmt.Sprintf("[%d, %d, %d]", entry.WorkgroupSize[0], entry.WorkgroupSize[1], entry.WorkgroupSize[2])},
		}
		if len(k.Module.Kernels) > 1 {
			var es []entryJSON
			for _, e := range k.Module.Kernels {
				es = append(es, entryJSON{Entry: e.Name, Workgroup: e.WorkgroupSize})
			}
			ms = append(ms, member{"entries", "Entries", "ElioEntry[]", jsonLit(es)})
		}
		bs, err := bindings(k.Module)
		if err != nil {
			return "", fmt.Errorf("ts: kernel %q: %w", k.Name, err)
		}
		ms = append(ms, member{"bindings", "Bindings", "ElioBinding[]", jsonLit(bs)})
		wu, su, differ, err := uniforms(k.Module)
		if err != nil {
			return "", fmt.Errorf("ts: kernel %q: %w", k.Name, err)
		}
		ms = append(ms, member{"uniforms", "Uniforms", "ElioUniform[]", jsonLit(wu)})
		if k.Gather && differ {
			ms = append(ms, member{"uniformsStd140", "UniformsStd140", "ElioUniform[]", jsonLit(su)})
		}
		if k.Gather {
			if errs := sema.CheckGather(k.Module, entry.Name); len(errs) > 0 {
				return "", fmt.Errorf("ts: kernel %q is not gather-compatible: %w", k.Name, errors.Join(errs...))
			}
			glsl, lay, err := gather.Emit(k.Module, entry.Name)
			if err != nil {
				return "", fmt.Errorf("ts: kernel %q: gather: %w", k.Name, err)
			}
			h.Write([]byte(glsl))
			ms = append(ms,
				member{"glsl", "GLSL", "", "`" + escapeTemplate(glsl) + "`"},
				member{"gatherLayout", "GatherLayout", "ElioGatherLayout", jsonLit(gatherLayout(lay))})
		}
		if meta.Global != "" {
			fmt.Fprintf(&body, "\n    // --- %s ---\n    %s: {\n", k.Name, stem)
			for _, m := range ms {
				fmt.Fprintf(&body, "      %s: %s,\n", m.key, m.lit)
			}
			body.WriteString("    },\n")
			continue
		}
		fmt.Fprintf(&body, "\n// --- %s ---\n\n", k.Name)
		for _, m := range ms {
			if m.typ != "" {
				fmt.Fprintf(&body, "export const %s%s: %s = %s;\n", stem, m.suffix, m.typ, m.lit)
			} else {
				fmt.Fprintf(&body, "export const %s%s = %s;\n", stem, m.suffix, m.lit)
			}
		}
	}
	sum := hex.EncodeToString(h.Sum(nil))
	var out strings.Builder
	fmt.Fprintf(&out, "// Code generated by elio emit ts; DO NOT EDIT.\n")
	fmt.Fprintf(&out, "// Elio %s, kernel set %q.\n", meta.Version, set.Name)
	fmt.Fprintf(&out, "// sha256: %s (of the WGSL and GLSL sources below, in order)\n", sum)
	if meta.Global == "" {
		out.WriteString(header)
		out.WriteString(body.String())
		return out.String(), nil
	}
	out.WriteString(globalHeader)
	fmt.Fprintf(&out, "  window.%s = deepFreeze({\n    elio: %s,\n    set: %s,\n    sha256: %q,\n    kernels: {\n", meta.Global, quote(meta.Version), quote(set.Name), sum)
	out.WriteString(body.String())
	out.WriteString("    },\n  });\n})();\n")
	return out.String(), nil
}

// globalHeader opens the classic-script form. Types are JSDoc, so the module
// carries no TypeScript syntax and a typed or untyped chunk can include it.
const globalHeader = `
/**
 * @typedef {{group: number, binding: number, name: string, kind: "uniform" | "storage-read" | "storage-rw" | "storage-texture" | "texture", format?: string, elementType: string}} ElioBinding
 * @typedef {{entry: string, workgroup: [number, number, number]}} ElioEntry
 * @typedef {{name: string, type: string, offset: number, size: number}} ElioField
 * @typedef {{group: number, binding: number, name: string, struct: string, size: number, fields: ElioField[]}} ElioUniform
 * @typedef {{kernel: string, inputs: {name: string, kind: "buffer" | "texture", elem: string, sampler: string, length: string, binding: number}[], outputs: {name: string, elem: string, format: string, location: number, variable: string}[], uniforms: {name: string, block: string, struct: string, binding: number}[]}} ElioGatherLayout
 */
(function () {
  "use strict";
  function deepFreeze(o) {
    if (o && typeof o === "object") {
      for (const key of Object.keys(o)) deepFreeze(o[key]);
      Object.freeze(o);
    }
    return o;
  }
`

const header = `
export interface ElioBinding {
  group: number;
  binding: number;
  name: string;
  kind: "uniform" | "storage-read" | "storage-rw" | "storage-texture" | "texture";
  format?: string;
  elementType: string;
}
export interface ElioEntry { entry: string; workgroup: [number, number, number]; }
export interface ElioField { name: string; type: string; offset: number; size: number; }
export interface ElioUniform {
  group: number;
  binding: number;
  name: string;
  struct: string;
  size: number;
  fields: ElioField[];
}
export interface ElioGatherLayout {
  kernel: string;
  inputs: { name: string; kind: "buffer" | "texture"; elem: string; sampler: string; length: string; binding: number }[];
  outputs: { name: string; elem: string; format: string; location: number; variable: string }[];
  uniforms: { name: string; block: string; struct: string; binding: number }[];
}
`

func primary(k Kernel) (ir.Kernel, error) {
	if k.Entry == "" {
		if len(k.Module.Kernels) == 1 {
			return k.Module.Kernels[0], nil
		}
		return ir.Kernel{}, fmt.Errorf("ts: kernel %q: module has %d entry points and no Entry is named", k.Name, len(k.Module.Kernels))
	}
	for _, e := range k.Module.Kernels {
		if e.Name == k.Entry {
			return e, nil
		}
	}
	return ir.Kernel{}, fmt.Errorf("ts: kernel %q: entry %q not found", k.Name, k.Entry)
}

func bindings(m *ir.Module) ([]bindingJSON, error) {
	out := []bindingJSON{}
	for _, b := range m.Bindings {
		j := bindingJSON{Group: b.Group, Binding: b.Binding, Name: b.Name}
		switch t := b.Type.(type) {
		case ir.StorageTexture:
			j.Kind, j.Format, j.ElementType = "storage-texture", t.Format, "vec4<f32>"
		case ir.Texture2D:
			j.Kind, j.ElementType = "texture", "vec4<f32>"
		case ir.Array:
			j.ElementType = typeString(t.Elem)
			j.Kind = storageKind(b)
		default:
			j.ElementType = typeString(b.Type)
			if b.Space == ir.Uniform {
				j.Kind = "uniform"
			} else {
				j.Kind = storageKind(b)
			}
		}
		if b.Space == ir.Uniform {
			j.Kind = "uniform"
		}
		out = append(out, j)
	}
	return out, nil
}

func storageKind(b ir.Binding) string {
	if b.Access == ir.Read {
		return "storage-read"
	}
	return "storage-rw"
}

func uniforms(m *ir.Module) (wg, std []uniformJSON, differ bool, err error) {
	wg, std = []uniformJSON{}, []uniformJSON{}
	for _, b := range m.Bindings {
		if b.Space != ir.Uniform {
			continue
		}
		n, ok := b.Type.(ir.Named)
		if !ok {
			return nil, nil, false, fmt.Errorf("uniform %s is not a struct", b.Name)
		}
		w, err := wgslLayout(m, n.Name)
		if err != nil {
			return nil, nil, false, err
		}
		s, err := std140Layout(m, n.Name)
		if err != nil {
			return nil, nil, false, err
		}
		wg = append(wg, toUniform(b, w))
		std = append(std, toUniform(b, s))
		if w.Size != s.Size {
			differ = true
		}
		for i := range w.Fields {
			if w.Fields[i].Offset != s.Fields[i].Offset {
				differ = true
			}
		}
	}
	return wg, std, differ, nil
}

func toUniform(b ir.Binding, l StructLayout) uniformJSON {
	u := uniformJSON{Group: b.Group, Binding: b.Binding, Name: b.Name, Struct: l.Struct, Size: l.Size, Fields: []fieldJSON{}}
	for _, f := range l.Fields {
		u.Fields = append(u.Fields, fieldJSON{f.Name, f.Type, f.Offset, f.Size})
	}
	return u
}

func gatherLayout(l gather.Layout) gatherLayoutJSON {
	j := gatherLayoutJSON{Kernel: l.Kernel, Inputs: []gatherBufferJSON{}, Outputs: []gatherOutputJSON{}, Uniforms: []gatherUniformJSON{}}
	for _, b := range l.Inputs {
		j.Inputs = append(j.Inputs, gatherBufferJSON{b.Name, b.Kind, b.Elem, b.Sampler, b.Length, b.Binding})
	}
	for _, o := range l.Outputs {
		j.Outputs = append(j.Outputs, gatherOutputJSON{o.Name, o.Elem, o.Format, o.Location, o.Variable})
	}
	for _, u := range l.Uniforms {
		j.Uniforms = append(j.Uniforms, gatherUniformJSON{u.Name, u.Block, u.Struct, u.Binding})
	}
	return j
}

// jsonLit renders v as indented JSON, which is also a TypeScript literal.
func jsonLit(v any) string {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		panic(err) // plain structs only
	}
	return strings.TrimRight(b.String(), "\n")
}

func quote(s string) string { return jsonLit(s) }

// escapeTemplate makes s safe inside a TypeScript template literal.
func escapeTemplate(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "`", "\\`")
	return strings.ReplaceAll(s, "${", "\\${")
}

func UnescapeTemplate(s string) string {
	s = strings.ReplaceAll(s, "\\${", "${")
	s = strings.ReplaceAll(s, "\\`", "`")
	return strings.ReplaceAll(s, "\\\\", "\\")
}

// camel turns snake_case into lowerCamelCase; a digit after an underscore
// stays a digit ("fft_gather_64" becomes "fftGather64").
func camel(s string) string {
	var b strings.Builder
	up := false
	for _, r := range s {
		if r == '_' {
			up = true
			continue
		}
		if up {
			r = unicode.ToUpper(r)
			up = false
		}
		b.WriteRune(r)
	}
	return b.String()
}
