package graph

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/sema"
)

// maxDispatch is WebGPU's default maxComputeWorkgroupsPerDimension.
const maxDispatch = 65535

var (
	frameSources = map[string]bool{"time": true, "dt": true, "cameraX": true, "cameraY": true, "cameraZ": true}
	exprOps      = map[string]int{"add": 2, "sub": 2, "mul": 2, "div": 2, "mod": 2, "min": 2, "max": 2, "floor": 1}
	bufferElems  = map[string]bool{"f32": true, "vec2<f32>": true, "vec4<f32>": true}
	texFormats   = map[string]bool{"rgba16float": true, "rgba32float": true, "r32float": true}
)

func numericType(t string) bool { return t == F32 || t == U32 || t == I32 }

// Validate checks g against the kernels it may use and returns every problem
// found, joined. The rules:
//
//  1. Every kernel binding is bound exactly once, to a resource of a compatible
//     kind, element type and format; every uniform block has one entry.
//  2. A pass never reads a resource it writes (the same resource and mip
//     level through two bindings); in-place read_write storage is allowed
//     only on WebGPU.
//  3. WebGL2 passes use only gather kernels, and every written resource is a
//     render target, at most 32 bytes per pixel in total.
//  4. Dispatch or target sizes cover every element of the written resources.
//  5. Every uniform field the kernel declares gets a value; every "param:"
//     reference (uniform, when, resource feed, expression) names a declared
//     param of a matching type; a param refers only to earlier params.
//  6. Every export names a texture resource (on WebGPU, a buffer too, for
//     readback).
func Validate(g Graph, kernels []Kernel) error {
	v := &validator{g: g, kernels: map[string]Kernel{}, params: map[string]Param{}, res: map[string]Resource{}}
	for _, k := range kernels {
		v.kernels[k.Name] = k
	}
	v.run()
	return errors.Join(v.errs...)
}

type validator struct {
	g       Graph
	kernels map[string]Kernel
	params  map[string]Param
	res     map[string]Resource
	errs    []error
}

func (v *validator) errf(format string, a ...any) {
	v.errs = append(v.errs, fmt.Errorf("graph %s/%s: %s", v.g.Backend, v.g.Tier, fmt.Sprintf(format, a...)))
}

func (v *validator) run() {
	g := v.g
	if g.Name == "" {
		v.errf("name is empty")
	}
	if g.Backend != WebGPU && g.Backend != WebGL2 {
		v.errf("backend %q is not webgpu or webgl2", g.Backend)
	}
	if g.Version != Version {
		v.errf("version %d, want %d", g.Version, Version)
	}
	v.checkParams()
	v.checkResources()
	passNames := map[string]bool{}
	for _, p := range g.Passes {
		if passNames[p.Name] {
			v.errf("duplicate pass name %q", p.Name)
		}
		passNames[p.Name] = true
		v.checkPass(p)
	}
	for _, s := range g.FrameSwaps {
		a, aok := v.res[s[0]]
		b, bok := v.res[s[1]]
		if !aok || !bok {
			v.errf("swap %v names an unknown resource", s)
			continue
		}
		a.Name, b.Name, a.Feed, b.Feed = "", "", "", ""
		if a != b {
			v.errf("swap %v: the two resources differ in kind, type or size", s)
		}
	}
	for name, r := range g.Exports {
		res, ok := v.res[r]
		switch {
		case !ok:
			v.errf("export %q names unknown resource %q", name, r)
		case res.Kind != "texture" && !(res.Kind == "buffer" && g.Backend == WebGPU):
			v.errf("export %q: %q must be a texture resource (a buffer only on webgpu)", name, r)
		}
	}
}

// refName returns the param name of a "param:<name>" reference.
func refName(s string) (string, bool) {
	n, ok := strings.CutPrefix(s, "param:")
	return n, ok && n != ""
}

func (v *validator) checkParams() {
	for _, p := range v.g.Params {
		if _, dup := v.params[p.Name]; dup || p.Name == "" {
			v.errf("param name %q is empty or duplicate", p.Name)
		}
		buffer := strings.HasSuffix(p.Type, "[]")
		switch {
		case buffer:
			if !bufferElems[strings.TrimSuffix(p.Type, "[]")] || p.Count <= 0 {
				v.errf("param %q: buffer type %q needs an element type and a count", p.Name, p.Type)
			}
		case !numericType(p.Type) && p.Type != Bool:
			v.errf("param %q: type %q is not f32, u32, i32, bool or a buffer", p.Name, p.Type)
		}
		s := p.Source
		if p.Tween < 0 || (p.Tween > 0 && (s.Kind != SourceValue || buffer)) {
			v.errf("param %q: tween applies only to numeric value params and cannot be negative", p.Name)
		}
		earlierNumeric := func(ref string) {
			n, ok := refName(ref)
			if !ok {
				v.errf("param %q: %q is not a param reference", p.Name, ref)
				return
			}
			q, ok := v.params[n]
			if !ok {
				v.errf("param %q: refers to %q, which is not an earlier param", p.Name, n)
			} else if !numericType(q.Type) {
				v.errf("param %q: %q is not numeric", p.Name, n)
			}
		}
		switch s.Kind {
		case SourceValue:
		case SourceFrame:
			if !frameSources[s.Name] {
				v.errf("param %q: unknown frame source %q", p.Name, s.Name)
			}
			if p.Type != F32 {
				v.errf("param %q: a frame source is f32", p.Name)
			}
		case SourceExpr:
			want, ok := exprOps[s.Op]
			if !ok {
				v.errf("param %q: unknown op %q", p.Name, s.Op)
			} else if len(s.Args) != want {
				v.errf("param %q: op %s takes %d args, got %d", p.Name, s.Op, want, len(s.Args))
			}
			for _, a := range s.Args {
				switch x := a.(type) {
				case string:
					earlierNumeric(x)
				default:
					if _, ok := number(a); !ok {
						v.errf("param %q: arg %v is not a number or a param reference", p.Name, a)
					}
				}
			}
			if !numericType(p.Type) {
				v.errf("param %q: an expression is numeric", p.Name)
			}
		case SourceAccumulate, SourceDelta:
			earlierNumeric(s.Of)
			if !numericType(p.Type) {
				v.errf("param %q: %s is numeric", p.Name, s.Kind)
			}
		default:
			v.errf("param %q: unknown source kind %q", p.Name, s.Kind)
		}
		v.params[p.Name] = p
	}
}

func (v *validator) checkResources() {
	g := v.g
	for _, r := range g.Resources {
		if _, dup := v.res[r.Name]; dup || r.Name == "" {
			v.errf("resource name %q is empty or duplicate", r.Name)
		}
		v.res[r.Name] = r
		switch r.Kind {
		case "buffer":
			if !bufferElems[r.ElementType] || r.Count <= 0 {
				v.errf("resource %q: buffer needs a known element type and a count", r.Name)
			}
			if g.Backend == WebGL2 && (r.Width <= 0 || r.Count%max(r.Width, 1) != 0) {
				v.errf("resource %q: a webgl2 buffer needs a width that divides its count", r.Name)
			}
		case "texture":
			if !texFormats[r.Format] || r.Width <= 0 || r.Height <= 0 {
				v.errf("resource %q: texture needs a known format and a size", r.Name)
			}
			maxLevels := 1
			for s := max(r.Width, r.Height); s > 1; s >>= 1 {
				maxLevels++
			}
			if r.MipLevels < 1 || r.MipLevels > maxLevels {
				v.errf("resource %q: %d mip levels, want 1 to %d", r.Name, r.MipLevels, maxLevels)
			}
		default:
			v.errf("resource %q: kind %q is not buffer or texture", r.Name, r.Kind)
		}
		if r.Feed != "" {
			p, ok := v.params[r.Feed]
			switch {
			case !ok || !strings.HasSuffix(p.Type, "[]"):
				v.errf("resource %q: feed %q is not a buffer param", r.Name, r.Feed)
			case strings.TrimSuffix(p.Type, "[]") != r.ElementType || p.Count > r.Count:
				v.errf("resource %q: feed %q does not fit (element type or count)", r.Name, r.Feed)
			}
		}
	}
}

func levelSize(r Resource, level int) (w, h int) {
	return max(r.Width>>level, 1), max(r.Height>>level, 1)
}

func (v *validator) checkWhen(pass, when string) {
	switch {
	case when == "always":
	case strings.HasPrefix(when, "param:"):
		n, _ := refName(when)
		if p, ok := v.params[n]; !ok {
			v.errf("pass %q: when refers to undeclared param %q", pass, n)
		} else if p.Type != Bool {
			v.errf("pass %q: when param %q is not bool", pass, n)
		}
	case strings.HasPrefix(when, "changed:"):
		list := strings.Split(strings.TrimPrefix(when, "changed:"), ",")
		for _, n := range list {
			if _, ok := v.params[n]; !ok {
				v.errf("pass %q: when refers to undeclared param %q", pass, n)
			}
		}
	default:
		v.errf("pass %q: when %q is not always, param:<name> or changed:<names>", pass, when)
	}
}

func (v *validator) checkPass(p Pass) {
	v.checkWhen(p.Name, p.When)
	switch p.Type {
	case "mipmap":
		r, ok := v.res[p.Resource]
		if v.g.Backend != WebGL2 {
			v.errf("pass %q: a mipmap pass is webgl2 only", p.Name)
		}
		if !ok || r.Kind != "texture" {
			v.errf("pass %q: mipmap needs a texture resource, got %q", p.Name, p.Resource)
		}
		return
	case "kernel":
	default:
		v.errf("pass %q: type %q is not kernel or mipmap", p.Name, p.Type)
		return
	}
	k, ok := v.kernels[p.Kernel]
	if !ok {
		v.errf("pass %q: unknown kernel %q", p.Name, p.Kernel)
		return
	}
	var entry *ir.Kernel
	for i := range k.Module.Kernels {
		if k.Module.Kernels[i].Name == p.Entry {
			entry = &k.Module.Kernels[i]
		}
	}
	if entry == nil {
		v.errf("pass %q: kernel %q has no entry %q", p.Name, p.Kernel, p.Entry)
		return
	}
	web2 := v.g.Backend == WebGL2
	if web2 {
		if !k.Gather || p.Entry != k.Entry {
			v.errf("pass %q: webgl2 runs gather kernels only; %q entry %q is not one", p.Name, p.Kernel, p.Entry)
		}
		if p.Dispatch != nil || p.Target == nil {
			v.errf("pass %q: a webgl2 pass has a target and no dispatch", p.Name)
		}
	} else if p.Target != nil || p.Dispatch == nil {
		v.errf("pass %q: a webgpu pass has a dispatch and no target", p.Name)
	}

	// Rule 1: bindings.
	byName := map[string]ir.Binding{}
	for _, b := range k.Module.Bindings {
		byName[b.Name] = b
	}
	bound := map[string]Binding{}
	for _, b := range p.Bindings {
		mb, ok := byName[b.Name]
		if !ok || mb.Space == ir.Uniform {
			v.errf("pass %q: kernel %q has no binding %q (uniforms go in uniforms)", p.Name, p.Kernel, b.Name)
			continue
		}
		if _, dup := bound[b.Name]; dup {
			v.errf("pass %q: binding %q is bound more than once", p.Name, b.Name)
			continue
		}
		bound[b.Name] = b
	}
	type use struct {
		binding      string
		read, writes bool
	}
	uses := map[string][]use{} // "resource@level" -> uses
	written := []struct {
		r     Resource
		level int
		b     ir.Binding
	}{}
	for _, mb := range k.Module.Bindings {
		if mb.Space == ir.Uniform {
			continue
		}
		b, ok := bound[mb.Name]
		if !ok {
			v.errf("pass %q: binding %q is not bound", p.Name, mb.Name)
			continue
		}
		r, ok := v.res[b.Resource]
		if !ok {
			v.errf("pass %q: binding %q: unknown resource %q", p.Name, mb.Name, b.Resource)
			continue
		}
		okType := true
		switch t := mb.Type.(type) {
		case ir.Array:
			if r.Kind != "buffer" {
				v.errf("pass %q: binding %q needs a buffer, %q is a %s", p.Name, mb.Name, r.Name, r.Kind)
				okType = false
			} else if want := typeString(t.Elem); r.ElementType != want {
				v.errf("pass %q: binding %q: element type %s, resource %q holds %s", p.Name, mb.Name, want, r.Name, r.ElementType)
				okType = false
			}
		case ir.StorageTexture, ir.Texture2D:
			if r.Kind != "texture" {
				v.errf("pass %q: binding %q needs a texture, %q is a %s", p.Name, mb.Name, r.Name, r.Kind)
				okType = false
			} else if st, ok := t.(ir.StorageTexture); ok && st.Format != r.Format {
				v.errf("pass %q: binding %q: format %s, resource %q is %s", p.Name, mb.Name, st.Format, r.Name, r.Format)
				okType = false
			}
		default:
			v.errf("pass %q: binding %q has a type the graph cannot bind", p.Name, mb.Name)
			okType = false
		}
		if !okType {
			continue
		}
		if r.Kind == "texture" && (b.Level < 0 || b.Level >= r.MipLevels) {
			v.errf("pass %q: binding %q: level %d, resource %q has %d", p.Name, mb.Name, b.Level, r.Name, r.MipLevels)
			continue
		}
		if r.Kind == "buffer" && b.Level != 0 {
			v.errf("pass %q: binding %q: a buffer has no levels", p.Name, mb.Name)
		}
		writes := mb.Access == ir.ReadWrite || mb.Access == ir.Write
		reads := mb.Access == ir.Read || (mb.Access == ir.ReadWrite && !web2)
		if mb.Access == ir.ReadWrite && web2 {
			reads = false // a gather kernel never reads a buffer it writes
		}
		key := fmt.Sprintf("%s@%d", r.Name, b.Level)
		uses[key] = append(uses[key], use{mb.Name, reads, writes})
		if writes {
			written = append(written, struct {
				r     Resource
				level int
				b     ir.Binding
			}{r, b.Level, mb})
		}
	}
	// Rule 2.
	for key, us := range uses {
		if len(us) < 2 {
			continue
		}
		anyWrite := false
		for _, u := range us {
			anyWrite = anyWrite || u.writes
		}
		if anyWrite {
			names := []string{}
			for _, u := range us {
				names = append(names, u.binding)
			}
			v.errf("pass %q: bindings %s share %s, which the pass reads and writes", p.Name, strings.Join(names, ", "), key)
		}
	}
	// Rules 3 and 4.
	if web2 && p.Target != nil {
		t := p.Target
		if t.Width <= 0 || t.Height <= 0 || t.ElioWidth != t.Width {
			v.errf("pass %q: target %dx%d with elioWidth %d; elioWidth must equal the width", p.Name, t.Width, t.Height, t.ElioWidth)
		}
		bytes := 0
		for _, w := range written {
			bytes += sema.GatherOutputBytes(w.b)
			var rw, rh int
			if w.r.Kind == "buffer" {
				rw, rh = w.r.Width, w.r.Count/max(w.r.Width, 1)
			} else {
				rw, rh = levelSize(w.r, w.level)
			}
			if rw != t.Width || rh != t.Height {
				v.errf("pass %q: output %q is %dx%d, target is %dx%d", p.Name, w.b.Name, rw, rh, t.Width, t.Height)
			}
		}
		if bytes > sema.MaxGatherOutputBytes || len(written) > sema.MaxGatherOutputs {
			v.errf("pass %q: %d outputs cost %d bytes per pixel; at most %d outputs and %d bytes", p.Name, len(written), bytes, sema.MaxGatherOutputs, sema.MaxGatherOutputBytes)
		}
	}
	if !web2 && p.Dispatch != nil {
		d := p.Dispatch
		inv := 1
		for i := 0; i < 3; i++ {
			if d[i] < 1 || d[i] > maxDispatch {
				v.errf("pass %q: dispatch %v out of range 1 to %d", p.Name, *d, maxDispatch)
				inv = 0
				break
			}
			inv *= d[i] * entry.WorkgroupSize[i]
		}
		per := 1
		if len(entry.Shared) > 0 {
			per = 2 // a workgroup-cooperative (FFT) lane handles two elements
		}
		for _, w := range written {
			n := w.r.Count
			if w.r.Kind == "texture" {
				tw, th := levelSize(w.r, w.level)
				n = tw * th
			}
			if inv*per < n {
				v.errf("pass %q: dispatch %v covers %d elements of %q, which has %d", p.Name, *d, inv*per, w.r.Name, n)
			}
		}
	}

	// Rule 5: uniforms.
	haveU := map[string]bool{}
	for _, u := range p.Uniforms {
		mb, ok := byName[u.Name]
		if !ok || mb.Space != ir.Uniform {
			v.errf("pass %q: kernel %q has no uniform %q", p.Name, p.Kernel, u.Name)
			continue
		}
		haveU[u.Name] = true
		v.checkUniform(p, k.Module, mb, u)
	}
	for _, mb := range k.Module.Bindings {
		if mb.Space == ir.Uniform && !haveU[mb.Name] {
			v.errf("pass %q: uniform %q has no entry", p.Name, mb.Name)
		}
	}
}

func (v *validator) checkUniform(p Pass, m *ir.Module, mb ir.Binding, u Uniform) {
	n, ok := mb.Type.(ir.Named)
	if !ok {
		v.errf("pass %q: uniform %q is not a struct", p.Name, u.Name)
		return
	}
	var st *ir.Struct
	for i := range m.Structs {
		if m.Structs[i].Name == n.Name {
			st = &m.Structs[i]
		}
	}
	if st == nil {
		v.errf("pass %q: struct %q not found", p.Name, n.Name)
		return
	}
	fields := map[string]string{}
	for _, f := range st.Fields {
		ft := typeString(f.Type)
		fields[f.Name] = ft
		val, ok := u.Values[f.Name]
		if !ok {
			v.errf("pass %q: uniform %q has no value for field %q", p.Name, u.Name, f.Name)
			continue
		}
		if s, isStr := val.(string); isStr {
			ref, ok := refName(s)
			if !ok {
				v.errf("pass %q: uniform %q field %q: %q is not a number or param reference", p.Name, u.Name, f.Name, s)
				continue
			}
			q, ok := v.params[ref]
			switch {
			case !ok:
				v.errf("pass %q: uniform %q field %q refers to undeclared param %q", p.Name, u.Name, f.Name, ref)
			case q.Type != ft:
				v.errf("pass %q: uniform %q field %q is %s, param %q has type %s", p.Name, u.Name, f.Name, ft, ref, q.Type)
			}
			continue
		}
		x, ok := number(val)
		switch {
		case !ok || math.IsNaN(x) || math.IsInf(x, 0):
			v.errf("pass %q: uniform %q field %q: %v is not a finite number", p.Name, u.Name, f.Name, val)
		case ft == U32 && (x < 0 || x != math.Trunc(x) || x > math.MaxUint32):
			v.errf("pass %q: uniform %q field %q: %v is not a u32", p.Name, u.Name, f.Name, val)
		case ft == I32 && (x != math.Trunc(x) || math.Abs(x) > math.MaxInt32):
			v.errf("pass %q: uniform %q field %q: %v is not an i32", p.Name, u.Name, f.Name, val)
		case ft != U32 && ft != I32 && ft != F32:
			v.errf("pass %q: uniform %q field %q has type %s; only scalar fields are supported", p.Name, u.Name, f.Name, ft)
		}
	}
	for name := range u.Values {
		if _, ok := fields[name]; !ok {
			v.errf("pass %q: uniform %q has no field %q", p.Name, u.Name, name)
		}
	}
}

func number(a any) (float64, bool) {
	switch x := a.(type) {
	case int:
		return float64(x), true
	case int32:
		return float64(x), true
	case int64:
		return float64(x), true
	case uint32:
		return float64(x), true
	case uint64:
		return float64(x), true
	case float32:
		return float64(x), true
	case float64:
		return x, true
	}
	return 0, false
}

func typeString(t ir.Type) string {
	switch x := t.(type) {
	case ir.Scalar:
		return x.Name
	case ir.Vec:
		return fmt.Sprintf("vec%d<%s>", x.N, x.Elem.Name)
	}
	return "?"
}

// MemoryBytes is the GPU memory the graph's resources take: buffers at their
// element size (a WebGL2 buffer is an RGBA32F texture, 16 bytes per element),
// textures with their mip chain.
func MemoryBytes(g Graph) int64 {
	var total int64
	for _, r := range g.Resources {
		if r.Kind == "buffer" {
			size := map[string]int64{"f32": 4, "vec2<f32>": 8, "vec4<f32>": 16}[r.ElementType]
			if g.Backend == WebGL2 {
				size = 16
			}
			total += size * int64(r.Count)
			continue
		}
		texel := map[string]int64{"rgba16float": 8, "rgba32float": 16, "r32float": 4}[r.Format]
		for l := 0; l < r.MipLevels; l++ {
			w, h := levelSize(r, l)
			total += texel * int64(w*h)
		}
	}
	return total
}
