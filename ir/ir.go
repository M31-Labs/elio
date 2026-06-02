// Package ir is Elio's imperative compute IR: the typed, backend-neutral
// representation of a render-coupled compute kernel.
//
// Unlike Selena's *total* expression IR (no loops, no mutation — deliberately,
// so presentation shaders are deterministic), Elio's IR is imperative by
// design: loops, mutable locals, control flow, buffer indexing, and atomics —
// because that is exactly what compute kernels (culling, simulation, scan/sort,
// skinning) require. Backend emitters (emit/wgsl today; emit/metal, emit/spirv,
// and a CPU fallback later) lower a Module to per-target source.
package ir

// Type is a shader value type. Concrete kinds implement it.
type Type interface{ isType() }

// Scalar is f32 / i32 / u32 / bool.
type Scalar struct{ Name string }

// Vec is an N-component vector of Elem (e.g. vec3<f32>).
type Vec struct {
	N    int
	Elem Scalar
}

// Mat is a matrix (e.g. mat4x4<f32>).
type Mat struct {
	Cols, Rows int
	Elem       Scalar
}

// Array is array<Elem, Len>; Len == 0 means a runtime-sized array.
type Array struct {
	Elem Type
	Len  int
}

// Atomic is atomic<Elem> (Elem is u32 or i32).
type Atomic struct{ Elem Scalar }

// Named refers to a declared Struct by name.
type Named struct{ Name string }

func (Scalar) isType() {}
func (Vec) isType()    {}
func (Mat) isType()    {}
func (Array) isType()  {}
func (Atomic) isType() {}
func (Named) isType()  {}

// Canonical scalars.
var (
	F32  = Scalar{"f32"}
	I32  = Scalar{"i32"}
	U32  = Scalar{"u32"}
	Bool = Scalar{"bool"}
)

// Field is one member of a Struct.
type Field struct {
	Name string
	Type Type
}

// Struct is a named aggregate type.
type Struct struct {
	Name   string
	Fields []Field
}

// AddressSpace is a binding's WGSL address space.
type AddressSpace string

const (
	Uniform AddressSpace = "uniform"
	Storage AddressSpace = "storage"
)

// Access is a storage binding's access mode.
type Access string

const (
	Read      Access = "read"
	ReadWrite Access = "read_write"
)

// Binding is a module-level resource (var<...>): a uniform block or storage
// buffer at a (group, binding) coordinate.
type Binding struct {
	Group, Binding int
	Space          AddressSpace
	Access         Access // only meaningful for Storage
	Name           string
	Type           Type
}

// Builtin is a compute entrypoint input bound to a WGSL @builtin (e.g.
// global_invocation_id).
type Builtin struct {
	Name    string // parameter name (e.g. "gid")
	Builtin string // builtin name (e.g. "global_invocation_id")
	Type    Type
}

// Shared is a workgroup-shared variable: one instance per workgroup, visible to
// every invocation in it (var<workgroup> in WGSL, shared in GLSL, threadgroup in
// Metal). It is what reductions, scans, and tile-based algorithms need — the
// state a workgroup cooperates through, paired with Barrier.
type Shared struct {
	Name string
	Type Type
}

// Kernel is a compute entrypoint with a workgroup size and a statement body.
type Kernel struct {
	Name          string
	WorkgroupSize [3]int
	Builtins      []Builtin
	Shared        []Shared // workgroup-shared declarations
	Body          []Stmt
}

// Const is a module-level compile-time constant: const Name : Type = Value;
type Const struct {
	Name  string
	Type  Type
	Value Expr
}

// Module is a complete compute unit: declared structs, module-level constants,
// bindings, and one or more kernels.
type Module struct {
	Structs  []Struct
	Consts   []Const
	Bindings []Binding
	Kernels  []Kernel
}

// --- Statements ---

// Span is a 1-based source position (the start of a construct). The zero value
// means "unknown" — hand-built IR and the recursive-descent parser leave it
// unset; the grammargen front-end populates it so diagnostics carry line:col.
type Span struct {
	Line, Col int
}

// IsZero reports whether the span is unknown.
func (s Span) IsZero() bool { return s.Line == 0 && s.Col == 0 }

// Stmt is one imperative statement.
type Stmt interface{ isStmt() }

// StmtSpan returns a statement's source position, or the zero Span if unknown.
func StmtSpan(s Stmt) Span {
	switch x := s.(type) {
	case Let:
		return x.Span
	case Var:
		return x.Span
	case Assign:
		return x.Span
	case If:
		return x.Span
	case For:
		return x.Span
	case While:
		return x.Span
	case Return:
		return x.Span
	case Break:
		return x.Span
	case Barrier:
		return x.Span
	case Do:
		return x.Span
	}
	return Span{}
}

// Let binds an immutable local: let Name = Value;
type Let struct {
	Name  string
	Value Expr
	Span  Span
}

// Var declares a mutable local: var Name [: Type] = Init;
type Var struct {
	Name string
	Type Type // optional; nil omits the annotation
	Init Expr
	Span Span
}

// Assign is Target = Value; or a compound assignment Target Op= Value. Op is ""
// for a plain "=", or one of "+","-","*","/","%" for "+=","-=","*=","/=","%=".
type Assign struct {
	Target Expr
	Value  Expr
	Op     string
	Span   Span
}

// If is if (Cond) { Then } [else { Else }].
type If struct {
	Cond Expr
	Then []Stmt
	Else []Stmt
	Span Span
}

// For is for (Init; Cond; Post) { Body }.
type For struct {
	Init Stmt
	Cond Expr
	Post Stmt
	Body []Stmt
	Span Span
}

// While is while (Cond) { Body }.
type While struct {
	Cond Expr
	Body []Stmt
	Span Span
}

// Return is return;
type Return struct{ Span Span }

// Break is break;
type Break struct{ Span Span }

// Do evaluates Expr for its side effect: Expr;
type Do struct {
	Expr Expr
	Span Span
}

// Barrier is a workgroup control barrier: every invocation in the workgroup must
// reach it before any proceeds, and shared writes before it are visible after.
// It must sit in uniform control flow (all invocations reach it).
type Barrier struct{ Span Span }

func (Let) isStmt()     {}
func (Var) isStmt()     {}
func (Assign) isStmt()  {}
func (If) isStmt()      {}
func (For) isStmt()     {}
func (While) isStmt()   {}
func (Return) isStmt()  {}
func (Break) isStmt()   {}
func (Do) isStmt()      {}
func (Barrier) isStmt() {}

// --- Expressions ---

// Expr is the imperative expression tree.
type Expr interface{ isExpr() }

// Name references a local, binding, or builtin parameter.
type Name struct{ Name string }

// Lit is a pre-spelled literal token (e.g. "true", "0", "1u", "6").
type Lit struct{ Text string }

// Binary is (L Op R).
type Binary struct {
	Op   string
	L, R Expr
}

// Unary is Op E (prefix, e.g. -x).
type Unary struct {
	Op string
	E  Expr
}

// Call is Func(Args...) — a builtin or stdlib call (dot, arrayLength, atomicAdd).
type Call struct {
	Func string
	Args []Expr
}

// Index is E[Idx].
type Index struct {
	E, Idx Expr
}

// Member is E.Field — field access or vector/matrix swizzle.
type Member struct {
	E     Expr
	Field string
}

// AddrOf is &E — a pointer, needed for atomic ops and arrayLength.
type AddrOf struct{ E Expr }

func (Name) isExpr()   {}
func (Lit) isExpr()    {}
func (Binary) isExpr() {}
func (Unary) isExpr()  {}
func (Call) isExpr()   {}
func (Index) isExpr()  {}
func (Member) isExpr() {}
func (AddrOf) isExpr() {}
