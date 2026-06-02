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
