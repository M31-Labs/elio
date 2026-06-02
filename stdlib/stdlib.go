// Package stdlib is Elio's library of reusable cooperative compute kernels — the
// parallel primitives a render-coupled engine is built from (reduce, scan,
// stream-compaction, sort). Each is returned as an ir.Module so callers compile
// it to any backend, run it on the CPU fallback, or wrap it as a GoSX
// ExternalComputePass. They are the building blocks for the M2 game-engine floor:
// particle compaction, Hi-Z occlusion, GPU-driven culling, and (later) the
// depth sort a Gaussian-splat pass needs.
//
// The kernels here operate over one workgroup-sized tile (64 lanes). Composing
// them across many tiles (a block-sum pass over per-tile totals) is a separate,
// higher-level step.
package stdlib

import "m31labs.dev/elio/ir"

// tileWidth is the workgroup size every primitive here scans/reduces over.
const tileWidth = 64

// Reduce returns a workgroup sum-reduction — one partial per 64-lane tile. It
// completes the primitive trio (Reduce, Scan, Compact); its implementation and
// execution test live with the reduction kernel in package ir.
func Reduce() *ir.Module { return ir.WorkgroupReduce() }

// Scan returns a workgroup inclusive prefix-sum (Hillis-Steele) over u32:
// output[i] = input[0] + … + input[i] within each 64-lane tile. It is the
// keystone primitive — stream compaction turns a predicate into output offsets
// with a scan, and radix sort scans digit histograms.
//
// Hillis-Steele is work-inefficient (O(n log n)) but branch-simple and a clean
// first cut; a Blelloch work-efficient scan can replace it behind the same
// signature later. Reads and writes to shared memory are separated by barriers
// in uniform control flow, so it is race-free (the lockstep CPU interpreter and
// -race confirm it).
func Scan() *ir.Module {
	lid := ir.Member{E: ir.Name{Name: "lid"}, Field: "x"}
	gidx := ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}
	t := ir.Name{Name: "t"}
	offset := ir.Name{Name: "offset"}
	tempAt := func(i ir.Expr) ir.Expr { return ir.Index{E: ir.Name{Name: "temp"}, Idx: i} }

	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "input", Type: ir.Array{Elem: ir.U32}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "output", Type: ir.Array{Elem: ir.U32}},
		},
		Kernels: []ir.Kernel{{
			Name:          "scan",
			WorkgroupSize: [3]int{tileWidth, 1, 1},
			Builtins: []ir.Builtin{
				{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
				{Name: "lid", Builtin: "local_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
			},
			Shared: []ir.Shared{{Name: "temp", Type: ir.Array{Elem: ir.U32, Len: tileWidth}}},
			Body: []ir.Stmt{
				ir.Let{Name: "t", Value: lid},
				ir.Assign{Target: tempAt(t), Value: ir.Index{E: ir.Name{Name: "input"}, Idx: gidx}},
				ir.Barrier{},
				ir.For{
					Init: ir.Var{Name: "offset", Type: ir.U32, Init: ir.Lit{Text: "1u"}},
					Cond: ir.Binary{Op: "<", L: offset, R: ir.Lit{Text: "64u"}},
					Post: ir.Assign{Target: offset, Value: ir.Binary{Op: "*", L: offset, R: ir.Lit{Text: "2u"}}},
					Body: []ir.Stmt{
						// Read the partner into a private temporary, then add it back
						// in a separate barrier-fenced phase so no lane reads a cell
						// another lane is concurrently writing.
						ir.Var{Name: "v", Type: ir.U32, Init: ir.Lit{Text: "0u"}},
						ir.If{
							Cond: ir.Binary{Op: ">=", L: t, R: offset},
							Then: []ir.Stmt{ir.Assign{Target: ir.Name{Name: "v"}, Value: tempAt(ir.Binary{Op: "-", L: t, R: offset})}},
						},
						ir.Barrier{},
						ir.If{
							Cond: ir.Binary{Op: ">=", L: t, R: offset},
							Then: []ir.Stmt{ir.Assign{Target: tempAt(t), Value: ir.Binary{Op: "+", L: tempAt(t), R: ir.Name{Name: "v"}}}},
						},
						ir.Barrier{},
					},
				},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "output"}, Idx: gidx}, Value: tempAt(t)},
			},
		}},
	}
}

// Sort returns a workgroup bitonic sort of a 64-lane tile of u32 keys into
// ascending order. It is the "sort" of the scan/sort/reduce/compact set — the
// primitive a Gaussian-splat pass needs for back-to-front depth ordering and a
// transparency pass needs for per-tile fragment sorting.
//
// Bitonic sort needs only compare-exchange, so it sidesteps Elio's lack of
// bitwise operators: the partner index t XOR j (j a power of two) is computed
// with integer divide/modulo — flipping bit log2(j) of t means +j when that bit
// is 0 and -j when it is 1. Only the lower lane of each pair writes, and a
// barrier fences every compare-exchange stage, so it is race-free.
func Sort() *ir.Module {
	u := func(s string) ir.Expr { return ir.Lit{Text: s} }
	nm := func(n string) ir.Expr { return ir.Name{Name: n} }
	bin := func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	dataAt := func(i ir.Expr) ir.Expr { return ir.Index{E: ir.Name{Name: "data"}, Idx: i} }
	gidx := ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}
	t, j, k, partner := nm("t"), nm("j"), nm("k"), nm("partner")
	a, b := nm("a"), nm("b")

	// bit m of x, where divisor = 2^m: (x / divisor) % 2
	bitSet := func(x, divisor ir.Expr, want string) ir.Expr {
		return bin("==", bin("%", bin("/", x, divisor), u("2u")), u(want))
	}
	// swap newT/newP (so the lower lane takes the larger key)
	swap := []ir.Stmt{
		ir.Assign{Target: nm("newT"), Value: b},
		ir.Assign{Target: nm("newP"), Value: a},
	}

	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "input", Type: ir.Array{Elem: ir.U32}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "output", Type: ir.Array{Elem: ir.U32}},
		},
		Kernels: []ir.Kernel{{
			Name:          "sort",
			WorkgroupSize: [3]int{tileWidth, 1, 1},
			Builtins: []ir.Builtin{
				{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
				{Name: "lid", Builtin: "local_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
			},
			Shared: []ir.Shared{{Name: "data", Type: ir.Array{Elem: ir.U32, Len: tileWidth}}},
			Body: []ir.Stmt{
				ir.Let{Name: "t", Value: ir.Member{E: ir.Name{Name: "lid"}, Field: "x"}},
				ir.Assign{Target: dataAt(t), Value: ir.Index{E: ir.Name{Name: "input"}, Idx: gidx}},
				ir.Barrier{},
				ir.For{ // k: size of the bitonic sequence being merged
					Init: ir.Var{Name: "k", Type: ir.U32, Init: u("2u")},
					Cond: bin("<=", k, u("64u")),
					Post: ir.Assign{Target: k, Value: bin("*", k, u("2u"))},
					Body: []ir.Stmt{
						ir.For{ // j: compare distance, halving
							Init: ir.Var{Name: "j", Type: ir.U32, Init: bin("/", k, u("2u"))},
							Cond: bin(">", j, u("0u")),
							Post: ir.Assign{Target: j, Value: bin("/", j, u("2u"))},
							Body: []ir.Stmt{
								// partner = t XOR j  (flip bit log2(j))
								ir.Var{Name: "partner", Type: ir.U32, Init: bin("+", t, j)},
								ir.If{Cond: bitSet(t, j, "1u"), Then: []ir.Stmt{ir.Assign{Target: partner, Value: bin("-", t, j)}}},
								// only the lower lane of each pair compare-exchanges
								ir.If{Cond: bin(">", partner, t), Then: []ir.Stmt{
									ir.Let{Name: "a", Value: dataAt(t)},
									ir.Let{Name: "b", Value: dataAt(partner)},
									ir.Var{Name: "newT", Type: ir.U32, Init: a},
									ir.Var{Name: "newP", Type: ir.U32, Init: b},
									// ascending when bit log2(k) of t is 0, else descending
									ir.If{
										Cond: bitSet(t, k, "0u"),
										Then: []ir.Stmt{ir.If{Cond: bin(">", a, b), Then: swap}},
										Else: []ir.Stmt{ir.If{Cond: bin("<", a, b), Then: swap}},
									},
									ir.Assign{Target: dataAt(t), Value: nm("newT")},
									ir.Assign{Target: dataAt(partner), Value: nm("newP")},
								}},
								ir.Barrier{},
							},
						},
					},
				},
				ir.Assign{Target: ir.Index{E: ir.Name{Name: "output"}, Idx: gidx}, Value: dataAt(t)},
			},
		}},
	}
}

// Compact returns a workgroup stream-compaction: it densely packs the nonzero
// elements of a 64-lane input tile into output, writing the surviving count to
// count[0]. It is the scan-driven, atomics-free generalization of the cull
// kernel — each lane's output slot is the exclusive prefix sum of the keep
// flags (offset = inclusive_scan - keep), so survivors land in stable order with
// no contention. Predicate here is "value != 0"; a real pass swaps in a
// frustum/occlusion test and carries records instead of scalars.
func Compact() *ir.Module {
	lid := ir.Member{E: ir.Name{Name: "lid"}, Field: "x"}
	gidx := ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}
	t := ir.Name{Name: "t"}
	offset := ir.Name{Name: "offset"}
	keep := ir.Name{Name: "keep"}
	val := ir.Name{Name: "val"}
	scanAt := func(i ir.Expr) ir.Expr { return ir.Index{E: ir.Name{Name: "scan"}, Idx: i} }
	u := func(s string) ir.Expr { return ir.Lit{Text: s} }

	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.Read, Name: "input", Type: ir.Array{Elem: ir.U32}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "output", Type: ir.Array{Elem: ir.U32}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.ReadWrite, Name: "count", Type: ir.Array{Elem: ir.U32}},
		},
		Kernels: []ir.Kernel{{
			Name:          "compact",
			WorkgroupSize: [3]int{tileWidth, 1, 1},
			Builtins: []ir.Builtin{
				{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
				{Name: "lid", Builtin: "local_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
			},
			Shared: []ir.Shared{{Name: "scan", Type: ir.Array{Elem: ir.U32, Len: tileWidth}}},
			Body: []ir.Stmt{
				ir.Let{Name: "t", Value: lid},
				ir.Let{Name: "val", Value: ir.Index{E: ir.Name{Name: "input"}, Idx: gidx}},
				// keep flag, seeded into the scan buffer
				ir.Var{Name: "keep", Type: ir.U32, Init: u("0u")},
				ir.If{Cond: ir.Binary{Op: "!=", L: val, R: u("0u")}, Then: []ir.Stmt{ir.Assign{Target: keep, Value: u("1u")}}},
				ir.Assign{Target: scanAt(t), Value: keep},
				ir.Barrier{},
				// inclusive scan of the keep flags (Hillis-Steele)
				ir.For{
					Init: ir.Var{Name: "offset", Type: ir.U32, Init: u("1u")},
					Cond: ir.Binary{Op: "<", L: offset, R: u("64u")},
					Post: ir.Assign{Target: offset, Value: ir.Binary{Op: "*", L: offset, R: u("2u")}},
					Body: []ir.Stmt{
						ir.Var{Name: "v", Type: ir.U32, Init: u("0u")},
						ir.If{Cond: ir.Binary{Op: ">=", L: t, R: offset},
							Then: []ir.Stmt{ir.Assign{Target: ir.Name{Name: "v"}, Value: scanAt(ir.Binary{Op: "-", L: t, R: offset})}}},
						ir.Barrier{},
						ir.If{Cond: ir.Binary{Op: ">=", L: t, R: offset},
							Then: []ir.Stmt{ir.Assign{Target: scanAt(t), Value: ir.Binary{Op: "+", L: scanAt(t), R: ir.Name{Name: "v"}}}}},
						ir.Barrier{},
					},
				},
				// survivors scatter to their exclusive-prefix slot
				ir.If{Cond: ir.Binary{Op: "==", L: keep, R: u("1u")},
					Then: []ir.Stmt{ir.Assign{
						Target: ir.Index{E: ir.Name{Name: "output"}, Idx: ir.Binary{Op: "-", L: scanAt(t), R: keep}},
						Value:  val,
					}}},
				// last lane writes the total kept
				ir.If{Cond: ir.Binary{Op: "==", L: t, R: u("63u")},
					Then: []ir.Stmt{ir.Assign{Target: ir.Index{E: ir.Name{Name: "count"}, Idx: u("0")}, Value: scanAt(u("63"))}}},
			},
		}},
	}
}
