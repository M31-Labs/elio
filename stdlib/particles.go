package stdlib

import "m31labs.dev/elio/ir"

// ParticleUpdate returns the core particle-simulation kernel — the per-particle
// integration step a renderer runs every frame. It ages each particle, respawns
// the expired ones at the emitter, and otherwise integrates velocity under
// gravity then advances position. It is the M2 "collapse the particle impls"
// move: one Elio kernel emits to WGSL/GLSL/Metal and runs on the CPU fallback,
// replacing the per-backend hand-written updates.
//
// The Particle struct uses eight scalar f32 fields rather than two vec4s, which
// is the *same 32-byte layout* as the engine's interleaved pos[4]+vel[4] record
// (px..age = pos.xyzw, vx..life = vel.xyzw), so it drops into the existing
// particle storage buffer — while staying within Elio's scalar surface (no
// vector constructors). Forces beyond gravity (drag, curl-noise) layer on later.
func ParticleUpdate() *ir.Module {
	i := ir.Name{Name: "i"}
	// particles[i].<field>
	pm := func(field string) ir.Expr {
		return ir.Member{E: ir.Index{E: ir.Name{Name: "particles"}, Idx: i}, Field: field}
	}
	// sim.<field>
	sm := func(field string) ir.Expr { return ir.Member{E: ir.Name{Name: "sim"}, Field: field} }
	bin := func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	set := func(field string, v ir.Expr) ir.Stmt { return ir.Assign{Target: pm(field), Value: v} }
	zero := ir.Lit{Text: "0.0"}

	scalarF32 := func(name string) ir.Field { return ir.Field{Name: name, Type: ir.F32} }

	return &ir.Module{
		Structs: []ir.Struct{
			{Name: "Sim", Fields: []ir.Field{
				scalarF32("dt"),
				scalarF32("gx"), scalarF32("gy"), scalarF32("gz"), // gravity
				scalarF32("ex"), scalarF32("ey"), scalarF32("ez"), // emitter origin
				scalarF32("ivx"), scalarF32("ivy"), scalarF32("ivz"), // respawn velocity
			}},
			{Name: "Particle", Fields: []ir.Field{
				scalarF32("px"), scalarF32("py"), scalarF32("pz"), scalarF32("age"),
				scalarF32("vx"), scalarF32("vy"), scalarF32("vz"), scalarF32("life"),
			}},
		},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "sim", Type: ir.Named{Name: "Sim"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "particles", Type: ir.Array{Elem: ir.Named{Name: "Particle"}}},
		},
		Kernels: []ir.Kernel{{
			Name:          "update",
			WorkgroupSize: [3]int{tileWidth, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body: []ir.Stmt{
				ir.Let{Name: "i", Value: ir.Member{E: ir.Name{Name: "gid"}, Field: "x"}},
				ir.Let{Name: "newAge", Value: bin("+", pm("age"), sm("dt"))},
				ir.If{
					Cond: bin(">=", ir.Name{Name: "newAge"}, pm("life")),
					Then: []ir.Stmt{ // respawn at the emitter with the configured velocity
						set("px", sm("ex")), set("py", sm("ey")), set("pz", sm("ez")),
						set("age", zero),
						set("vx", sm("ivx")), set("vy", sm("ivy")), set("vz", sm("ivz")),
					},
					Else: []ir.Stmt{ // integrate velocity under gravity, then position
						ir.Let{Name: "nvx", Value: bin("+", pm("vx"), bin("*", sm("gx"), sm("dt")))},
						ir.Let{Name: "nvy", Value: bin("+", pm("vy"), bin("*", sm("gy"), sm("dt")))},
						ir.Let{Name: "nvz", Value: bin("+", pm("vz"), bin("*", sm("gz"), sm("dt")))},
						set("vx", ir.Name{Name: "nvx"}), set("vy", ir.Name{Name: "nvy"}), set("vz", ir.Name{Name: "nvz"}),
						set("px", bin("+", pm("px"), bin("*", ir.Name{Name: "nvx"}, sm("dt")))),
						set("py", bin("+", pm("py"), bin("*", ir.Name{Name: "nvy"}, sm("dt")))),
						set("pz", bin("+", pm("pz"), bin("*", ir.Name{Name: "nvz"}, sm("dt")))),
						set("age", ir.Name{Name: "newAge"}),
					},
				},
			},
		}},
	}
}
