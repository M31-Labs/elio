package stdlib

import "m31labs.dev/elio/ir"

// GalaxyParticleUpdate is the galaxy star/particle integrator authored once in
// Elio — the per-frame compute step that drives m31labs.dev's galaxy. It is the
// single-source replacement for GoSX's hand-written
// render/bundle/particles.go:particleUpdateWGSL: it keeps the exact buffer
// layout (Particle = position vec4 [xyz + age], velocity vec4 [xyz + lifetime];
// ParticleForce = cfg vec4 [kind, strength, frequency, _], vector vec4) and
// the same force graph (gravity, drag, wind, attractor, vortex/orbit,
// turbulence) so the Elio-emitted WGSL drops into the existing storage buffer
// and uniform, while the same source also yields GLSL, Metal, and a CPU
// fallback for backends without compute.
//
// Force kinds (cfg.x): 1 gravity, 2 drag, 3 wind, 4 attractor, 5 vortex,
// 6 turbulence — matching the host force encoding.
func GalaxyParticleUpdate() *ir.Module {
	vec4 := ir.Vec{N: 4, Elem: ir.F32}

	// expression helpers
	name := func(s string) ir.Expr { return ir.Name{Name: s} }
	lit := func(s string) ir.Expr { return ir.Lit{Text: s} }
	mem := func(e ir.Expr, f string) ir.Expr { return ir.Member{E: e, Field: f} }
	mem2 := func(e ir.Expr, a, b string) ir.Expr { return ir.Member{E: ir.Member{E: e, Field: a}, Field: b} }
	idx := func(e, i ir.Expr) ir.Expr { return ir.Index{E: e, Idx: i} }
	call := func(fn string, a ...ir.Expr) ir.Expr { return ir.Call{Func: fn, Args: a} }
	bin := func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	u := func(f string) ir.Expr { return mem(name("u"), f) }
	u2 := func(a, b string) ir.Expr { return mem2(name("u"), a, b) }
	p := func(f string) ir.Expr { return mem(name("p"), f) }
	p2 := func(a, b string) ir.Expr { return mem2(name("p"), a, b) }
	force2 := func(a, b string) ir.Expr { return mem2(name("force"), a, b) }
	// inRange builds (k > lo) && (k < hi) — selects a force kind without a cast.
	inRange := func(k ir.Expr, lo, hi string) ir.Expr {
		return bin("&&", bin(">", k, lit(lo)), bin("<", k, lit(hi)))
	}

	// statement helpers
	set := func(t, v ir.Expr) ir.Stmt { return ir.Assign{Target: t, Value: v} }
	addTo := func(t, v ir.Expr) ir.Stmt { return ir.Assign{Target: t, Op: "+", Value: v} }
	letS := func(n string, v ir.Expr) ir.Stmt { return ir.Let{Name: n, Value: v} }

	pos := name("pos")
	accel := name("acceleration")
	strength := name("strength")
	fv := name("fv")
	kind := name("kind")

	// One loop iteration: read force fi, branch on kind, accumulate.
	forceBody := []ir.Stmt{
		letS("force", idx(u("forces"), name("fi"))),
		letS("kind", force2("cfg", "x")),
		letS("strength", force2("cfg", "y")),
		letS("fv", force2("vector", "xyz")),
		// drag (2): accumulate the scalar drag coefficient.
		ir.If{Cond: inRange(kind, "1.5", "2.5"), Then: []ir.Stmt{addTo(name("drag"), strength)}},
		// gravity (1) and wind (3): a constant directional acceleration.
		ir.If{Cond: inRange(kind, "0.5", "1.5"), Then: []ir.Stmt{addTo(accel, bin("*", fv, strength))}},
		ir.If{Cond: inRange(kind, "2.5", "3.5"), Then: []ir.Stmt{addTo(accel, bin("*", fv, strength))}},
		// attractor (4): pull toward the target point fv.
		ir.If{Cond: inRange(kind, "3.5", "4.5"), Then: []ir.Stmt{
			letS("delta", bin("-", fv, pos)),
			addTo(accel, bin("*", call("normalize", name("delta")), strength)),
		}},
		// vortex / orbit (5): tangential force around axis fv.
		ir.If{Cond: inRange(kind, "4.5", "5.5"), Then: []ir.Stmt{
			letS("axis", call("normalize", fv)),
			letS("radial", bin("-", pos, bin("*", name("axis"), call("dot", pos, name("axis"))))),
			addTo(accel, bin("*", call("normalize", call("cross", name("radial"), name("axis"))), strength)),
		}},
		// turbulence (6): a curl-ish sin/cos field at frequency cfg.z.
		ir.If{Cond: inRange(kind, "5.5", "6.5"), Then: []ir.Stmt{
			letS("freq", call("abs", force2("cfg", "z"))),
			letS("nx", bin("*",
				call("sin", bin("+", bin("*", mem(pos, "x"), name("freq")), bin("*", u("time"), lit("1.3")))),
				call("cos", bin("+", bin("*", mem(pos, "z"), name("freq")), bin("*", u("time"), lit("0.7")))))),
			letS("ny", bin("*",
				call("sin", bin("+", bin("*", mem(pos, "y"), name("freq")), bin("*", u("time"), lit("0.9")))),
				call("cos", bin("+", bin("*", mem(pos, "x"), name("freq")), bin("*", u("time"), lit("1.1")))))),
			letS("nz", bin("*",
				call("sin", bin("+", bin("*", mem(pos, "z"), name("freq")), bin("*", u("time"), lit("1.7")))),
				call("cos", bin("+", bin("*", mem(pos, "y"), name("freq")), bin("*", u("time"), lit("0.5")))))),
			addTo(accel, bin("*", call("vec3", name("nx"), name("ny"), name("nz")), strength)),
		}},
	}

	integrate := []ir.Stmt{
		letS("pos", p2("position", "xyz")),
		letS("vel", p2("velocity", "xyz")),
		ir.Var{Name: "acceleration", Init: call("vec3", lit("0.0"), lit("0.0"), lit("0.0"))},
		ir.Var{Name: "drag", Init: lit("0.0")},
		ir.For{
			Init: ir.Var{Name: "fi", Type: ir.I32, Init: lit("0")},
			Cond: bin("<", call("f32", name("fi")), u("forceCount")),
			Post: ir.Assign{Target: name("fi"), Value: bin("+", name("fi"), lit("1"))},
			Body: forceBody,
		},
		letS("dragFactor", call("clamp", bin("-", lit("1.0"), bin("*", name("drag"), u("dt"))), lit("0.0"), lit("1.0"))),
		letS("newVel", bin("+", bin("*", name("vel"), name("dragFactor")), bin("*", accel, u("dt")))),
		letS("newPos", bin("+", pos, bin("*", name("newVel"), u("dt")))),
		set(p("position"), call("vec4", name("newPos"), name("newAge"))),
		set(p("velocity"), call("vec4", name("newVel"), p2("velocity", "w"))),
	}

	// Respawn: hash the index+time for a pseudo-random direction, emit at the
	// emitter position offset by the emitter radius, with the configured speed.
	hash := func(seed string) ir.Expr {
		return call("fract", bin("*", call("sin", bin("+", bin("*", name("fid"), lit(seed)), u("time"))), lit("43758.5453")))
	}
	respawn := []ir.Stmt{
		letS("fid", call("f32", name("i"))),
		letS("ha", hash("12.9898")),
		letS("hb", hash("78.233")),
		letS("hc", hash("37.719")),
		letS("dir", call("normalize", call("vec3",
			bin("-", bin("*", name("ha"), lit("2.0")), lit("1.0")),
			bin("+", bin("*", name("hb"), lit("0.4")), lit("0.3")),
			bin("-", bin("*", name("hc"), lit("2.0")), lit("1.0"))))),
		letS("off", bin("*", call("vec3",
			bin("-", bin("*", name("ha"), lit("2.0")), lit("1.0")),
			bin("-", bin("*", name("hb"), lit("2.0")), lit("1.0")),
			bin("-", bin("*", name("hc"), lit("2.0")), lit("1.0"))), u2("emitterPos", "w"))),
		set(p("position"), call("vec4", bin("+", u2("emitterPos", "xyz"), name("off")), lit("0.0"))),
		set(p("velocity"), call("vec4", bin("*", name("dir"), u2("initialSpeed", "x")), u("lifetime"))),
	}

	body := []ir.Stmt{
		letS("i", mem(name("gid"), "x")),
		ir.If{Cond: bin(">=", name("i"), call("arrayLength", ir.AddrOf{E: name("particles")})), Then: []ir.Stmt{ir.Return{}}},
		ir.Var{Name: "p", Init: idx(name("particles"), name("i"))},
		letS("newAge", bin("+", p2("position", "w"), u("dt"))),
		ir.If{
			Cond: bin("||", bin(">=", name("newAge"), p2("velocity", "w")), bin("<=", p2("velocity", "w"), lit("0.0"))),
			Then: respawn,
			Else: integrate,
		},
		set(idx(name("particles"), name("i")), name("p")),
	}

	field := func(n string, t ir.Type) ir.Field { return ir.Field{Name: n, Type: t} }
	return &ir.Module{
		Structs: []ir.Struct{
			{Name: "Particle", Fields: []ir.Field{field("position", vec4), field("velocity", vec4)}},
			{Name: "ParticleForce", Fields: []ir.Field{field("cfg", vec4), field("vector", vec4)}},
			{Name: "ParticleUniforms", Fields: []ir.Field{
				field("dt", ir.F32), field("time", ir.F32), field("lifetime", ir.F32), field("forceCount", ir.F32),
				field("emitterPos", vec4), field("initialSpeed", vec4),
				field("forces", ir.Array{Elem: ir.Named{Name: "ParticleForce"}, Len: 8}),
			}},
		},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Uniform, Name: "u", Type: ir.Named{Name: "ParticleUniforms"}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "particles", Type: ir.Array{Elem: ir.Named{Name: "Particle"}}},
		},
		Kernels: []ir.Kernel{{
			Name:          "update",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body:          body,
		}},
	}
}
