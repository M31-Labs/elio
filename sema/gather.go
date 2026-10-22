package sema

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"m31labs.dev/elio/ir"
)

// MaxGatherOutputs is the number of color attachments a WebGL2 pass can write.
const MaxGatherOutputs = 4

// CheckGather reports why kernel cannot run as a fullscreen fragment pass over
// float textures (the WebGL2 "gather" form). It returns one diagnostic per
// violation of the gather profile, each naming the kernel and the rule:
//
//  1. no shared memory, barriers or atomics;
//  2. the only builtin is global_invocation_id;
//  3. every storage-buffer write is `buf[i] = value`, with i = gid.x or a let
//     bound directly to gid.x;
//  4. at most four buffers are written; each is read_write and never read;
//  5. buffer elements are f32, vec2<f32> or vec4<f32>;
//  6. uniform blocks hold only scalars (f32, i32, u32) and vectors of them;
//  7. a written StorageTexture is an output when every textureStore to it uses
//     the coordinate vec2u(i % W, i / W), with i = gid.x (or a let bound to it)
//     and W a uniform field or textureDimensions(t).x; W must equal the pass
//     width (elio_width), so each invocation writes its own pixel;
//  8. all outputs (buffers and textures) together cost at most 32 bytes per
//     pixel: a buffer or RGBA32F texture 16, RGBA16F 8, R32F 4.
//
// Buffer and texture outputs share the MaxGatherOutputs limit of rule 4.
//
// Only bindings the kernel uses are checked. CheckGather does not run Check.
func CheckGather(m *ir.Module, kernel string) []error {
	var k *ir.Kernel
	for i := range m.Kernels {
		if m.Kernels[i].Name == kernel {
			k = &m.Kernels[i]
			break
		}
	}
	if k == nil {
		return []error{fmt.Errorf("gather: kernel %q not found", kernel)}
	}
	g := &gatherCheck{m: m, k: k, bindings: map[string]ir.Binding{}}
	for _, b := range m.Bindings {
		g.bindings[b.Name] = b
	}
	g.run()
	return g.errs
}

type gatherCheck struct {
	m        *ir.Module
	k        *ir.Kernel
	bindings map[string]ir.Binding
	errs     []error

	gidName string          // name of the global_invocation_id parameter
	idxLets map[string]bool // lets bound directly to gid.x
	used    map[string]bool // bindings referenced by the kernel
	read    map[string]bool // storage buffers read
	written map[string]bool // storage buffers written
	texOut  map[string]bool // storage textures written by textureStore
	lets    map[string]ir.Expr
}

// MaxGatherOutputBytes is the per-pixel byte budget of a WebGL2 pass's color
// attachments (MAX_COLOR_ATTACHMENT_BYTES_PER_PIXEL guarantees 32).
const MaxGatherOutputBytes = 32

// GatherOutputBytes is the per-pixel cost of one output binding: 16 for a
// buffer or an RGBA32F texture, 8 for RGBA16F, 4 for R32F.
func GatherOutputBytes(b ir.Binding) int {
	if t, ok := b.Type.(ir.StorageTexture); ok {
		switch t.Format {
		case "rgba16float":
			return 8
		case "r32float":
			return 4
		}
	}
	return 16
}

func (g *gatherCheck) errf(rule int, format string, a ...any) {
	g.errs = append(g.errs, fmt.Errorf("gather: kernel %q: rule %d: %s", g.k.Name, rule, fmt.Sprintf(format, a...)))
}

func (g *gatherCheck) run() {
	g.used, g.read, g.written = map[string]bool{}, map[string]bool{}, map[string]bool{}
	g.texOut = map[string]bool{}

	// Rule 1: shared memory.
	for _, sh := range g.k.Shared {
		g.errf(1, "shared memory %q is not allowed (no workgroup cooperation in a fragment pass)", sh.Name)
	}
	// Rule 2: builtins.
	for _, bi := range g.k.Builtins {
		if bi.Builtin != "global_invocation_id" {
			g.errf(2, "builtin %q (%s) is not allowed; only global_invocation_id", bi.Name, bi.Builtin)
		} else {
			g.gidName = bi.Name
		}
	}
	g.collectIndexLets()

	walkStmts(g.k.Body, func(s ir.Stmt) {
		switch x := s.(type) {
		case ir.Barrier:
			g.errf(1, "barrier is not allowed")
		case ir.Var:
			if _, ok := x.Type.(ir.Atomic); ok {
				g.errf(1, "atomic local %q is not allowed", x.Name)
			}
		case ir.Assign:
			g.checkAssignTarget(x)
		case ir.Do:
			if c, ok := x.Expr.(ir.Call); ok && c.Func == "textureStore" && len(c.Args) == 3 {
				if n, ok := c.Args[0].(ir.Name); ok && g.isTexture(n.Name) {
					g.checkTextureStore(n.Name, c)
					g.visitReads(c.Args[1])
					g.visitReads(c.Args[2])
					return
				}
			}
		}
		for _, e := range stmtExprs(s) {
			g.visitReads(e)
		}
	})

	// Rule 4.
	var names []string
	for n := range g.written {
		names = append(names, n)
	}
	sort.Strings(names)
	var texNames []string
	for n := range g.texOut {
		texNames = append(texNames, n)
	}
	sort.Strings(texNames)
	if len(names)+len(texNames) > MaxGatherOutputs {
		g.errf(4, "%d outputs are written (%s); at most %d", len(names)+len(texNames), strings.Join(append(append([]string{}, names...), texNames...), ", "), MaxGatherOutputs)
	}
	bytes := 0
	for _, n := range append(append([]string{}, names...), texNames...) {
		bytes += GatherOutputBytes(g.bindings[n])
	}
	if bytes > MaxGatherOutputBytes {
		g.errf(8, "outputs cost %d bytes per pixel (%s); at most %d bytes per pixel", bytes, strings.Join(append(append([]string{}, names...), texNames...), ", "), MaxGatherOutputBytes)
	}
	for _, n := range names {
		if b := g.bindings[n]; b.Access != ir.ReadWrite {
			g.errf(4, "written buffer %q must be read_write", n)
		}
		if g.read[n] {
			g.errf(4, "buffer %q is both written and read", n)
		}
	}

	// Rules 5 and 6 over the bindings the kernel uses, in declaration order.
	for _, b := range g.m.Bindings {
		if !g.used[b.Name] {
			continue
		}
		switch b.Space {
		case ir.Storage:
			if _, ok := b.Type.(ir.StorageTexture); ok {
				if !g.texOut[b.Name] {
					g.errf(3, "texture %q is used but never written with textureStore; a gather texture is an output only", b.Name)
				}
				continue
			}
			g.checkBufferType(b)
		case ir.Uniform:
			g.checkUniform(b)
		}
	}
}

// collectIndexLets finds names that are bound only to gid.x.
func (g *gatherCheck) collectIndexLets() {
	g.idxLets = map[string]bool{}
	g.lets = map[string]ir.Expr{}
	other := map[string]bool{}
	seen := map[string]bool{}
	walkStmts(g.k.Body, func(s ir.Stmt) {
		switch x := s.(type) {
		case ir.Let:
			if seen[x.Name] {
				delete(g.lets, x.Name)
				other[x.Name] = true
			} else {
				g.lets[x.Name] = x.Value
			}
			seen[x.Name] = true
			if g.isGidX(x.Value) {
				g.idxLets[x.Name] = true
			} else {
				other[x.Name] = true
			}
		case ir.Var:
			other[x.Name] = true
			delete(g.lets, x.Name)
		}
	})
	for n := range other {
		delete(g.idxLets, n)
	}
}

func (g *gatherCheck) isGidX(e ir.Expr) bool {
	m, ok := e.(ir.Member)
	if !ok || m.Field != "x" || g.gidName == "" {
		return false
	}
	n, ok := m.E.(ir.Name)
	return ok && n.Name == g.gidName
}

func (g *gatherCheck) isIndexOK(e ir.Expr) bool {
	if g.isGidX(e) {
		return true
	}
	n, ok := e.(ir.Name)
	return ok && g.idxLets[n.Name]
}

func (g *gatherCheck) isStorage(name string) bool {
	b, ok := g.bindings[name]
	if ok {
		if _, tex := b.Type.(ir.StorageTexture); tex {
			return false // textures are outputs, tracked apart from buffers
		}
	}
	return ok && b.Space == ir.Storage
}

func (g *gatherCheck) isTexture(name string) bool {
	b, ok := g.bindings[name]
	if !ok {
		return false
	}
	_, tex := b.Type.(ir.StorageTexture)
	return tex
}

// resolveLet follows a name bound once by let to its value.
func (g *gatherCheck) resolveLet(e ir.Expr) ir.Expr {
	for i := 0; i < 8; i++ {
		n, ok := e.(ir.Name)
		if !ok {
			return e
		}
		v, ok := g.lets[n.Name]
		if !ok {
			return e
		}
		e = v
	}
	return e
}

// isWidth reports whether e is a uniform field or textureDimensions(tex).x.
func (g *gatherCheck) isWidth(e ir.Expr, tex string) bool {
	m, ok := e.(ir.Member)
	if !ok {
		return false
	}
	switch b := m.E.(type) {
	case ir.Name:
		bd, ok := g.bindings[b.Name]
		return ok && bd.Space == ir.Uniform
	case ir.Call:
		if b.Func != "textureDimensions" || m.Field != "x" || len(b.Args) != 1 {
			return false
		}
		n, ok := b.Args[0].(ir.Name)
		return ok && n.Name == tex
	}
	return false
}

// checkTextureStore requires the coordinate vec2u(i % W, i / W).
func (g *gatherCheck) checkTextureStore(tex string, c ir.Call) {
	g.used[tex] = true
	g.texOut[tex] = true
	if b := g.bindings[tex]; b.Access != ir.Write {
		g.errf(3, "texture %q must be a write binding", tex)
	}
	if g.coordOK(tex, c.Args[1]) {
		return
	}
	g.errf(3, "textureStore to %q must use the coordinate vec2u(i %% W, i / W) with i = gid.x (or a let bound to it) and W a uniform field or textureDimensions(%s).x", tex, tex)
}

func (g *gatherCheck) coordOK(tex string, e ir.Expr) bool {
	c, ok := g.resolveLet(e).(ir.Call)
	if !ok || len(c.Args) != 2 || (c.Func != "vec2u" && c.Func != "vec2<u32>") {
		return false
	}
	part := func(a ir.Expr, op string) (ir.Expr, bool) {
		b, ok := g.resolveLet(a).(ir.Binary)
		if !ok || b.Op != op || !g.isIndexOK(b.L) || !g.isWidth(g.resolveLet(b.R), tex) {
			return nil, false
		}
		return g.resolveLet(b.R), true
	}
	w0, ok0 := part(c.Args[0], "%")
	w1, ok1 := part(c.Args[1], "/")
	return ok0 && ok1 && reflect.DeepEqual(w0, w1)
}

func (g *gatherCheck) checkAssignTarget(a ir.Assign) {
	root, _ := rootName(a.Target)
	if !g.isStorage(root) {
		return
	}
	g.used[root] = true
	g.written[root] = true
	ix, ok := a.Target.(ir.Index)
	isName := false
	if ok {
		_, isName = ix.E.(ir.Name)
	}
	switch {
	case !ok || !isName:
		g.errf(3, "write to %q must be %s[i] = value; field and swizzle writes into an element are not allowed", root, root)
	case a.Op != "":
		g.errf(3, "compound assignment (%s=) to %q reads the buffer; write %s[i] = value", a.Op, root, root)
	case !g.isIndexOK(ix.Idx):
		g.errf(3, "write to %q must index with gid.x or a let bound directly to gid.x", root)
	}
}

// visitReads records bindings used and storage buffers read in e. The write
// target base of an Assign is not passed here (see stmtExprs).
func (g *gatherCheck) visitReads(e ir.Expr) {
	dims, dimsX := 0, 0
	walkExpr(e, func(x ir.Expr) {
		switch v := x.(type) {
		case ir.Member:
			if c, ok := v.E.(ir.Call); ok && c.Func == "textureDimensions" && v.Field == "x" {
				dimsX++
			}
		case ir.Name:
			if _, ok := g.bindings[v.Name]; ok {
				g.used[v.Name] = true
				if g.isStorage(v.Name) {
					g.read[v.Name] = true
				}
			}
		case ir.Call:
			if strings.HasPrefix(v.Func, "atomic") {
				g.errf(1, "atomic call %s is not allowed", v.Func)
			}
			if v.Func == "textureDimensions" {
				dims++
			}
		}
	})
	if dims != dimsX {
		g.errf(3, "textureDimensions is only allowed as textureDimensions(t).x, the pass width; texture %q", firstTexture(g, e))
	}
}

// firstTexture names a texture referenced by e, for diagnostics.
func firstTexture(g *gatherCheck, e ir.Expr) string {
	name := ""
	walkExpr(e, func(x ir.Expr) {
		if n, ok := x.(ir.Name); ok && name == "" && g.isTexture(n.Name) {
			name = n.Name
		}
	})
	return name
}

func (g *gatherCheck) checkBufferType(b ir.Binding) {
	arr, ok := b.Type.(ir.Array)
	if !ok {
		g.errf(5, "buffer %q must be an array of f32, vec2<f32> or vec4<f32>", b.Name)
		return
	}
	switch el := arr.Elem.(type) {
	case ir.Atomic:
		g.errf(1, "atomic buffer %q is not allowed", b.Name)
	case ir.Scalar:
		if el != ir.F32 {
			g.errf(5, "buffer %q has element type %s; need f32, vec2<f32> or vec4<f32>", b.Name, el.Name)
		}
	case ir.Vec:
		if el.Elem != ir.F32 || (el.N != 2 && el.N != 4) {
			g.errf(5, "buffer %q has element type vec%d<%s>; need f32, vec2<f32> or vec4<f32>", b.Name, el.N, el.Elem.Name)
		}
	case ir.Named:
		g.errf(5, "buffer %q has struct element type %s; structs are not allowed in gather buffers", b.Name, el.Name)
	default:
		g.errf(5, "buffer %q has an unsupported element type; need f32, vec2<f32> or vec4<f32>", b.Name)
	}
}

func (g *gatherCheck) checkUniform(b ir.Binding) {
	n, ok := b.Type.(ir.Named)
	if !ok {
		g.errf(6, "uniform %q must be a struct of scalars and vectors", b.Name)
		return
	}
	for _, s := range g.m.Structs {
		if s.Name != n.Name {
			continue
		}
		for _, f := range s.Fields {
			if !uniformFieldOK(f.Type) {
				g.errf(6, "uniform %q field %q has a type outside scalars (f32, i32, u32) and vectors of them", b.Name, f.Name)
			}
		}
		return
	}
	g.errf(6, "uniform %q has unknown struct %q", b.Name, n.Name)
}

func uniformFieldOK(t ir.Type) bool {
	switch x := t.(type) {
	case ir.Scalar:
		return x == ir.F32 || x == ir.U32 || x == ir.I32
	case ir.Vec:
		return x.Elem == ir.F32 || x.Elem == ir.U32 || x.Elem == ir.I32
	}
	return false
}

// stmtExprs returns the expressions a statement evaluates directly. For an
// Assign the target is included only through its index expressions (the base
// buffer name of a write is not a read).
func stmtExprs(s ir.Stmt) []ir.Expr {
	switch x := s.(type) {
	case ir.Let:
		return []ir.Expr{x.Value}
	case ir.Var:
		return []ir.Expr{x.Init}
	case ir.Assign:
		out := []ir.Expr{x.Value}
		t := x.Target
	peel:
		for {
			switch y := t.(type) {
			case ir.Index:
				out = append(out, y.Idx)
				t = y.E
			case ir.Member:
				t = y.E
			default:
				break peel
			}
		}
		if n, ok := t.(ir.Name); ok && x.Op != "" {
			out = append(out, n) // compound assignment reads its target
		}
		return out
	case ir.If:
		return []ir.Expr{x.Cond}
	case ir.For:
		return []ir.Expr{x.Cond}
	case ir.While:
		return []ir.Expr{x.Cond}
	case ir.Do:
		return []ir.Expr{x.Expr}
	}
	return nil
}

func walkStmts(body []ir.Stmt, fn func(ir.Stmt)) {
	for _, s := range body {
		fn(s)
		switch x := s.(type) {
		case ir.If:
			walkStmts(x.Then, fn)
			walkStmts(x.Else, fn)
		case ir.For:
			if x.Init != nil {
				walkStmts([]ir.Stmt{x.Init}, fn)
			}
			if x.Post != nil {
				walkStmts([]ir.Stmt{x.Post}, fn)
			}
			walkStmts(x.Body, fn)
		case ir.While:
			walkStmts(x.Body, fn)
		}
	}
}

func walkExpr(e ir.Expr, fn func(ir.Expr)) {
	if e == nil {
		return
	}
	fn(e)
	switch x := e.(type) {
	case ir.Binary:
		walkExpr(x.L, fn)
		walkExpr(x.R, fn)
	case ir.Unary:
		walkExpr(x.E, fn)
	case ir.Call:
		for _, a := range x.Args {
			walkExpr(a, fn)
		}
	case ir.Index:
		walkExpr(x.E, fn)
		walkExpr(x.Idx, fn)
	case ir.Member:
		walkExpr(x.E, fn)
	case ir.AddrOf:
		walkExpr(x.E, fn)
	}
}
