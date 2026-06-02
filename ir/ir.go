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

// Module is a complete compute unit: declared structs, module bindings, and one
// or more kernels.
type Module struct {
	Structs  []Struct
	Bindings []Binding
	Kernels  []Kernel
}

// --- Statements ---

// Stmt is one imperative statement.
type Stmt interface{ isStmt() }

// Let binds an immutable local: let Name = Value;
type Let struct {
	Name  string
	Value Expr
}

// Var declares a mutable local: var Name [: Type] = Init;
type Var struct {
	Name string
	Type Type // optional; nil omits the annotation
	Init Expr
}

// Assign is Target = Value;
type Assign struct {
	Target Expr
	Value  Expr
}

// If is if (Cond) { Then } [else { Else }].
type If struct {
	Cond Expr
	Then []Stmt
	Else []Stmt
}

// For is for (Init; Cond; Post) { Body }.
type For struct {
	Init Stmt
	Cond Expr
	Post Stmt
	Body []Stmt
}

// Return is return;
type Return struct{}

// Break is break;
type Break struct{}

// Do evaluates Expr for its side effect: Expr;
type Do struct{ Expr Expr }

// Barrier is a workgroup control barrier: every invocation in the workgroup must
// reach it before any proceeds, and shared writes before it are visible after.
// It must sit in uniform control flow (all invocations reach it).
type Barrier struct{}

func (Let) isStmt()     {}
func (Var) isStmt()     {}
func (Assign) isStmt()  {}
func (If) isStmt()      {}
func (For) isStmt()     {}
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
