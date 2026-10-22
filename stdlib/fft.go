package stdlib

import (
	"fmt"
	"math/bits"

	"m31labs.dev/elio/ir"
)

// Inverse 2D FFT kernels, in a compute form (FFT2DCompute) and a per-stage
// gather form (FFT2DGather).
//
// Contract shared by both forms:
//   - n is a power of two, 4..256.
//   - Data is one array<vec4<f32>> of n*n elements, row-major (index z*n + x).
//     Each element holds two complex numbers (re0, im0, re1, im1), so one pass
//     transforms two signals.
//   - The transform is the inverse, unnormalized: out[x] = sum_k in[k]*e^{+2*pi*i*k*x/n}.
//     Callers scale. Input and output are in natural order.
//   - Twiddles are computed inline with cos and sin of 2*pi*j/n.
//
// Stockham autosort (radix-2, decimation in frequency). N is the line length
// and L = log2 N. At stage t (0 <= t < L) the sub-transform length is
// len = N >> t and the stride is s = 1 << t. Each of the N/2 butterflies has an
// index j in [0, N/2):
//
//	p = j / s        (0 <= p < len/2)
//	q = j % s        (0 <= q < s)
//	a = x[j]         (= x[q + s*p])
//	b = x[j + N/2]   (= x[q + s*(p + len/2)])
//	w = e^{+2*pi*i*(p*s)/N}
//	y[q + s*(2p)]     = a + b          (= y[j + s*p])
//	y[q + s*(2p + 1)] = (a - b) * w    (= y[j + s*p + s])
//
// The output of one stage is the input of the next, ping-ponging between two
// buffers. After stage L-1 the result is in natural order, so no bit-reversal
// pass is needed. The gather form inverts the write map: output index o splits
// as q = o % s, u = o / s, r = u % 2, p = u / 2, and reads a and b at
// j = p*s + q.

// fftSize checks n and returns log2 n.
func fftSize(n int) (int, error) {
	if n < 4 || n > 256 || n&(n-1) != 0 {
		return 0, fmt.Errorf("stdlib: FFT size %d: need a power of two from 4 to 256", n)
	}
	return bits.TrailingZeros(uint(n)), nil
}

const fftTwoPi = "6.283185307179586"

// fftHelpers are small IR builders local to the FFT kernels.
var (
	fu   = func(s string) ir.Expr { return ir.Lit{Text: s} }
	fnm  = func(s string) ir.Expr { return ir.Name{Name: s} }
	fbin = func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	fvec = ir.Vec{N: 4, Elem: ir.F32}
)

func fcall(f string, a ...ir.Expr) ir.Expr { return ir.Call{Func: f, Args: a} }
func fmem(e ir.Expr, f string) ir.Expr     { return ir.Member{E: e, Field: f} }
func fidx(e, i ir.Expr) ir.Expr            { return ir.Index{E: e, Idx: i} }

// fftTwiddled returns the IR for (d * w) where d is the packed difference
// vec4 (two complex numbers) and w = cos(angle) + i sin(angle).
func fftTwiddled(d, angle ir.Expr) []ir.Stmt {
	c, s := fnm("wc"), fnm("ws")
	dn := fnm("d")
	return []ir.Stmt{
		ir.Let{Name: "wc", Value: fcall("cos", angle)},
		ir.Let{Name: "ws", Value: fcall("sin", angle)},
		ir.Let{Name: "d", Value: d},
		ir.Let{Name: "tw", Value: fcall("vec4",
			fbin("-", fbin("*", fmem(dn, "x"), c), fbin("*", fmem(dn, "y"), s)),
			fbin("+", fbin("*", fmem(dn, "x"), s), fbin("*", fmem(dn, "y"), c)),
			fbin("-", fbin("*", fmem(dn, "z"), c), fbin("*", fmem(dn, "w"), s)),
			fbin("+", fbin("*", fmem(dn, "z"), s), fbin("*", fmem(dn, "w"), c)),
		)},
	}
}

// FFT2DGather returns the per-stage gather form: one kernel, fft_stage, with
// one invocation per element writing only its own index. The host runs
// 2*log2(n) passes (stage 0..log2n-1 for axis 0, then again for axis 1),
// swapping src and dst between passes.
//
//	struct FFTStage { n: u32; stage: u32; axis: u32; _pad: u32; }  // axis 0 = rows, 1 = columns
//	@group(0) @binding(0) uniform fft: FFTStage;
//	@group(0) @binding(1) storage read src: []vec4<f32>;
//	@group(0) @binding(2) storage read_write dst: []vec4<f32>;
//
// Dispatch n*n invocations (workgroup size 64; extra lanes return).
func FFT2DGather(n int) (*ir.Module, error) {
	if _, err := fftSize(n); err != nil {
		return nil, err
	}
	fftN := func(f string) ir.Expr { return fmem(fnm("fft"), f) }
	idx, nn := fnm("idx"), fnm("n")
	line, pos, ls, ps := fnm("line"), fnm("pos"), fnm("ls"), fnm("ps")
	at := func(j ir.Expr) ir.Expr { return fidx(fnm("src"), fbin("+", fbin("*", line, ls), fbin("*", j, ps))) }
	angle := fbin("/", fbin("*", fu(fftTwoPi), fcall("f32", fbin("*", fnm("p"), fnm("s")))), fcall("f32", nn))
	elseBody := append(fftTwiddled(fbin("-", fnm("a"), fnm("b")), angle),
		ir.Assign{Target: fidx(fnm("dst"), idx), Value: fnm("tw")})

	body := []ir.Stmt{
		ir.Let{Name: "idx", Value: fmem(fnm("gid"), "x")},
		ir.Let{Name: "n", Value: fftN("n")},
		ir.If{Cond: fbin(">=", idx, fbin("*", nn, nn)), Then: []ir.Stmt{ir.Return{}}},
		// rows: line = z, pos = x; columns: line = x, pos = z (address = line*ls + pos*ps).
		ir.Var{Name: "line", Type: ir.U32, Init: fbin("/", idx, nn)},
		ir.Var{Name: "pos", Type: ir.U32, Init: fbin("%", idx, nn)},
		ir.Var{Name: "ls", Type: ir.U32, Init: nn},
		ir.Var{Name: "ps", Type: ir.U32, Init: fu("1u")},
		ir.If{Cond: fbin("==", fftN("axis"), fu("1u")), Then: []ir.Stmt{
			ir.Assign{Target: line, Value: fbin("%", idx, nn)},
			ir.Assign{Target: pos, Value: fbin("/", idx, nn)},
			ir.Assign{Target: ls, Value: fu("1u")},
			ir.Assign{Target: ps, Value: nn},
		}},
		ir.Let{Name: "s", Value: fbin("<<", fu("1u"), fftN("stage"))},
		ir.Let{Name: "q", Value: fbin("%", pos, fnm("s"))},
		ir.Let{Name: "u", Value: fbin("/", pos, fnm("s"))},
		ir.Let{Name: "r", Value: fbin("%", fnm("u"), fu("2u"))},
		ir.Let{Name: "p", Value: fbin("/", fnm("u"), fu("2u"))},
		ir.Let{Name: "j", Value: fbin("+", fbin("*", fnm("p"), fnm("s")), fnm("q"))},
		ir.Let{Name: "a", Value: at(fnm("j"))},
		ir.Let{Name: "b", Value: at(fbin("+", fnm("j"), fbin("/", nn, fu("2u"))))},
		ir.If{
			Cond: fbin("==", fnm("r"), fu("0u")),
			Then: []ir.Stmt{ir.Assign{Target: fidx(fnm("dst"), idx), Value: fbin("+", fnm("a"), fnm("b"))}},
			Else: elseBody,
		},
	}
	return &ir.Module{
		Structs: []ir.Struct{{Name: "FFTStage", Fields: []ir.Field{
			{Name: "n", Type: ir.U32}, {Name: "stage", Type: ir.U32}, {Name: "axis", Type: ir.U32}, {Name: "_pad", Type: ir.U32},
		}}},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "fft", Type: ir.Named{Name: "FFTStage"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.Read, Name: "src", Type: ir.Array{Elem: fvec}},
			{Group: 0, Binding: 2, Space: ir.Storage, Access: ir.ReadWrite, Name: "dst", Type: ir.Array{Elem: fvec}},
		},
		Kernels: []ir.Kernel{{
			Name:          "fft_stage",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body:          body,
		}},
	}, nil
}

// FFT2DCompute returns the workgroup form: two kernels, fft_rows and
// fft_cols, each dispatched as n workgroups of n/2 lanes. Workgroup wid.x
// transforms row wid.x (fft_rows) or column wid.x (fft_cols) entirely in
// shared memory: it loads the line, runs log2(n) Stockham stages (the formula
// is in the package comment above) with a workgroup barrier between stages,
// and writes the line back. Shared memory is array<vec4<f32>, 2n>, two halves
// that ping-pong: stage t reads half t%2 and writes half (t+1)%2. Each lane
// handles butterfly j = lid.x, so one dispatch covers all n/2 butterflies.
//
//	@group(0) @binding(0) storage read_write data: []vec4<f32>;
//
// Run fft_rows, then fft_cols, with a storage barrier between the dispatches.
func FFT2DCompute(n int) (*ir.Module, error) {
	log2n, err := fftSize(n)
	if err != nil {
		return nil, err
	}
	lit := func(v int) ir.Expr { return fu(fmt.Sprintf("%du", v)) }
	half := n / 2
	j := fnm("j")
	sh := func(i ir.Expr) ir.Expr { return fidx(fnm("buf"), i) }
	data := func(i ir.Expr) ir.Expr { return fidx(fnm("data"), i) }

	kernel := func(name string, addr func(pos ir.Expr) ir.Expr) ir.Kernel {
		angle := fbin("/", fbin("*", fu(fftTwoPi), fcall("f32", fbin("*", fnm("p"), fnm("s")))), fcall("f32", lit(n)))
		stage := append([]ir.Stmt{
			ir.Let{Name: "s", Value: fbin("<<", fu("1u"), fnm("t"))},
			ir.Let{Name: "p", Value: fbin("/", j, fnm("s"))},
			ir.Let{Name: "rd", Value: fbin("*", fbin("%", fnm("t"), fu("2u")), lit(n))},
			ir.Let{Name: "wr", Value: fbin("*", fbin("%", fbin("+", fnm("t"), fu("1u")), fu("2u")), lit(n))},
			ir.Let{Name: "a", Value: sh(fbin("+", fnm("rd"), j))},
			ir.Let{Name: "b", Value: sh(fbin("+", fbin("+", fnm("rd"), j), lit(half)))},
			// y[j + s*p] = a + b; y[j + s*p + s] = (a - b) * w
			ir.Let{Name: "o", Value: fbin("+", fnm("wr"), fbin("+", j, fbin("*", fnm("s"), fnm("p"))))},
			ir.Assign{Target: sh(fnm("o")), Value: fbin("+", fnm("a"), fnm("b"))},
		}, fftTwiddled(fbin("-", fnm("a"), fnm("b")), angle)...)
		stage = append(stage,
			ir.Assign{Target: sh(fbin("+", fnm("o"), fnm("s"))), Value: fnm("tw")},
			ir.Barrier{})
		final := fbin("*", fu(fmt.Sprintf("%du", log2n%2)), lit(n))
		return ir.Kernel{
			Name:          name,
			WorkgroupSize: [3]int{half, 1, 1},
			Builtins: []ir.Builtin{
				{Name: "lid", Builtin: "local_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
				{Name: "wid", Builtin: "workgroup_id", Type: ir.Vec{N: 3, Elem: ir.U32}},
			},
			Shared: []ir.Shared{{Name: "buf", Type: ir.Array{Elem: fvec, Len: 2 * n}}},
			Body: []ir.Stmt{
				ir.Let{Name: "j", Value: fmem(fnm("lid"), "x")},
				ir.Let{Name: "line", Value: fmem(fnm("wid"), "x")},
				ir.Assign{Target: sh(j), Value: data(addr(j))},
				ir.Assign{Target: sh(fbin("+", j, lit(half))), Value: data(addr(fbin("+", j, lit(half))))},
				ir.Barrier{},
				ir.For{
					Init: ir.Var{Name: "t", Type: ir.U32, Init: fu("0u")},
					Cond: fbin("<", fnm("t"), lit(log2n)),
					Post: ir.Assign{Target: fnm("t"), Value: fbin("+", fnm("t"), fu("1u"))},
					Body: stage,
				},
				ir.Assign{Target: data(addr(j)), Value: sh(fbin("+", final, j))},
				ir.Assign{Target: data(addr(fbin("+", j, lit(half)))), Value: sh(fbin("+", fbin("+", final, j), lit(half)))},
			},
		}
	}
	line := fnm("line")
	return &ir.Module{
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "data", Type: ir.Array{Elem: fvec}},
		},
		Kernels: []ir.Kernel{
			kernel("fft_rows", func(pos ir.Expr) ir.Expr { return fbin("+", fbin("*", line, lit(n)), pos) }),
			kernel("fft_cols", func(pos ir.Expr) ir.Expr { return fbin("+", fbin("*", pos, lit(n)), line) }),
		},
	}, nil
}
