package stdlib

import "m31labs.dev/elio/ir"

// galaxyExprHelpers bundles the tiny expression-building closures so
// galaxy_physics helpers can be called from both galaxy.go (native kernel) and
// galaxy_browser.go (browser kernel) without duplicating the helpers.
type galaxyExprHelpers struct {
	name  func(string) ir.Expr
	lit   func(string) ir.Expr
	mem   func(ir.Expr, string) ir.Expr
	call  func(string, ...ir.Expr) ir.Expr
	bin   func(string, ir.Expr, ir.Expr) ir.Expr
	set   func(ir.Expr, ir.Expr) ir.Stmt
	addTo func(ir.Expr, ir.Expr) ir.Stmt
	letS  func(string, ir.Expr) ir.Stmt
}

// newExprHelpers constructs a galaxyExprHelpers populated with the standard
// Elio IR primitives.
func newExprHelpers() galaxyExprHelpers {
	h := galaxyExprHelpers{}
	h.name = func(s string) ir.Expr { return ir.Name{Name: s} }
	h.lit = func(s string) ir.Expr { return ir.Lit{Text: s} }
	h.mem = func(e ir.Expr, f string) ir.Expr { return ir.Member{E: e, Field: f} }
	h.call = func(fn string, a ...ir.Expr) ir.Expr { return ir.Call{Func: fn, Args: a} }
	h.bin = func(op string, l, r ir.Expr) ir.Expr { return ir.Binary{Op: op, L: l, R: r} }
	h.set = func(t, v ir.Expr) ir.Stmt { return ir.Assign{Target: t, Value: v} }
	h.addTo = func(t, v ir.Expr) ir.Stmt { return ir.Assign{Target: t, Op: "+", Value: v} }
	h.letS = func(n string, v ir.Expr) ir.Stmt { return ir.Let{Name: n, Value: v} }
	return h
}

// browserForceLoop builds the shared force-accumulation loop body for the
// browser-contract kernel. The browser kernel stores force kind as u32 (integer
// switch) and uses scalar particle fields (posX/posY/posZ/velX/velY/velZ)
// instead of vec4 position/velocity.
//
// The loop body reads forces[fi], dispatches on kind (0=gravity, 1=wind,
// 2=turbulence, 3=orbit, 4=drag, 5=radial), accumulates velocity delta into
// dvX/dvY/dvZ local vars, then adds them onto velX/velY/velZ.  This exactly
// mirrors the browser WGSL kernel's inline apply* helper calls.
//
// Force struct fields: kind u32, strength f32, dirX/dirY/dirZ f32,
// frequency f32, _pad0/_pad1 f32.
//
// Parameters: the names of the particle scalar fields and the params binding,
// and totalTimeExpr to pass to turbulence.
func browserForceLoop(
	h galaxyExprHelpers,
	posX, posY, posZ, velX, velY, velZ ir.Expr,
	forceCountExpr ir.Expr, // params.forceCount (u32)
	dtExpr ir.Expr, // params.deltaTime
	totalTimeExpr ir.Expr, // params.totalTime
	forcesBinding ir.Expr, // forces array expression
) []ir.Stmt {
	v3 := func(x, y, z ir.Expr) ir.Expr { return h.call("vec3f", x, y, z) }
	// Force field accessors (f is a local Force struct value).
	fKind := h.mem(h.name("bfl_f"), "kind")
	fStr := h.mem(h.name("bfl_f"), "strength")
	fDirX := h.mem(h.name("bfl_f"), "dirX")
	fDirY := h.mem(h.name("bfl_f"), "dirY")
	fDirZ := h.mem(h.name("bfl_f"), "dirZ")
	fFreq := h.mem(h.name("bfl_f"), "frequency")

	// Helper: append the dv onto velX/Y/Z accumulators.
	applyDV := []ir.Stmt{
		ir.Assign{Target: velX, Op: "+", Value: h.mem(h.name("bfl_dv"), "x")},
		ir.Assign{Target: velY, Op: "+", Value: h.mem(h.name("bfl_dv"), "y")},
		ir.Assign{Target: velZ, Op: "+", Value: h.mem(h.name("bfl_dv"), "z")},
	}

	makeDV := func(x, y, z ir.Expr) ir.Stmt {
		return h.letS("bfl_dv", v3(x, y, z))
	}

	// gravity (kind==0): dir * strength * dt
	gravityBody := append(
		[]ir.Stmt{makeDV(
			h.bin("*", h.bin("*", fDirX, fStr), dtExpr),
			h.bin("*", h.bin("*", fDirY, fStr), dtExpr),
			h.bin("*", h.bin("*", fDirZ, fStr), dtExpr),
		)},
		applyDV...,
	)

	// wind (kind==1): identical to gravity
	windBody := append(
		[]ir.Stmt{makeDV(
			h.bin("*", h.bin("*", fDirX, fStr), dtExpr),
			h.bin("*", h.bin("*", fDirY, fStr), dtExpr),
			h.bin("*", h.bin("*", fDirZ, fStr), dtExpr),
		)},
		applyDV...,
	)

	// turbulence (kind==2): sin/cos noise field
	turbBody := []ir.Stmt{
		h.letS("bfl_freq", fFreq),
		h.letS("bfl_nx", h.bin("*",
			h.call("sin", h.bin("+", h.bin("*", posX, h.name("bfl_freq")), h.bin("*", totalTimeExpr, h.lit("1.3")))),
			h.call("cos", h.bin("+", h.bin("*", posZ, h.name("bfl_freq")), h.bin("*", totalTimeExpr, h.lit("0.7")))))),
		h.letS("bfl_ny", h.bin("*",
			h.call("sin", h.bin("+", h.bin("*", posY, h.name("bfl_freq")), h.bin("*", totalTimeExpr, h.lit("0.9")))),
			h.call("cos", h.bin("+", h.bin("*", posX, h.name("bfl_freq")), h.bin("*", totalTimeExpr, h.lit("1.1")))))),
		h.letS("bfl_nz", h.bin("*",
			h.call("sin", h.bin("+", h.bin("*", posZ, h.name("bfl_freq")), h.bin("*", totalTimeExpr, h.lit("1.7")))),
			h.call("cos", h.bin("+", h.bin("*", posY, h.name("bfl_freq")), h.bin("*", totalTimeExpr, h.lit("0.5")))))),
		makeDV(
			h.bin("*", h.bin("*", h.name("bfl_nx"), fStr), dtExpr),
			h.bin("*", h.bin("*", h.name("bfl_ny"), fStr), dtExpr),
			h.bin("*", h.bin("*", h.name("bfl_nz"), fStr), dtExpr),
		),
	}
	turbBody = append(turbBody, applyDV...)

	// orbit (kind==3): tangential around Y axis in XZ plane.
	// dx=posX, dz=posZ; dist=max(sqrt(dx*dx+dz*dz),0.001);
	// dv = (-dz/dist, 0, dx/dist) * strength * dt
	orbitBody := []ir.Stmt{
		h.letS("bfl_dx", posX),
		h.letS("bfl_dz", posZ),
		h.letS("bfl_dist", h.call("max",
			h.call("sqrt", h.bin("+",
				h.bin("*", h.name("bfl_dx"), h.name("bfl_dx")),
				h.bin("*", h.name("bfl_dz"), h.name("bfl_dz")))),
			h.lit("0.001"))),
		makeDV(
			h.bin("*", h.bin("*", h.bin("/", h.bin("*", h.lit("-1.0"), h.name("bfl_dz")), h.name("bfl_dist")), fStr), dtExpr),
			h.bin("*", h.bin("*", h.lit("0.0"), fStr), dtExpr),
			h.bin("*", h.bin("*", h.bin("/", h.name("bfl_dx"), h.name("bfl_dist")), fStr), dtExpr),
		),
	}
	orbitBody = append(orbitBody, applyDV...)

	// drag (kind==4): -vel * strength * dt
	dragBody := append(
		[]ir.Stmt{makeDV(
			h.bin("*", h.bin("*", h.bin("*", h.lit("-1.0"), velX), fStr), dtExpr),
			h.bin("*", h.bin("*", h.bin("*", h.lit("-1.0"), velY), fStr), dtExpr),
			h.bin("*", h.bin("*", h.bin("*", h.lit("-1.0"), velZ), fStr), dtExpr),
		)},
		applyDV...,
	)

	// radial (kind==5): outward from origin (with optional bias).
	// dir = (posX, posY, posZ)
	// if biasLen > 0.001: dir += normalize(bias) * max(length(dir), 1.0) * 0.35
	// dist = max(length(dir), 0.001); dv = (dir / dist) * strength * dt
	radialBody := []ir.Stmt{
		ir.Var{Name: "bfl_rx", Init: posX},
		ir.Var{Name: "bfl_ry", Init: posY},
		ir.Var{Name: "bfl_rz", Init: posZ},
		h.letS("bfl_biasLen", h.call("sqrt", h.bin("+", h.bin("+",
			h.bin("*", fDirX, fDirX),
			h.bin("*", fDirY, fDirY)),
			h.bin("*", fDirZ, fDirZ)))),
		ir.If{
			Cond: h.bin(">", h.name("bfl_biasLen"), h.lit("0.001")),
			Then: []ir.Stmt{
				h.letS("bfl_radialLen", h.call("max",
					h.call("sqrt", h.bin("+", h.bin("+",
						h.bin("*", h.name("bfl_rx"), h.name("bfl_rx")),
						h.bin("*", h.name("bfl_ry"), h.name("bfl_ry"))),
						h.bin("*", h.name("bfl_rz"), h.name("bfl_rz")))),
					h.lit("1.0"))),
				ir.Assign{Target: h.name("bfl_rx"), Op: "+", Value: h.bin("*", h.bin("*", h.bin("/", fDirX, h.name("bfl_biasLen")), h.name("bfl_radialLen")), h.lit("0.35"))},
				ir.Assign{Target: h.name("bfl_ry"), Op: "+", Value: h.bin("*", h.bin("*", h.bin("/", fDirY, h.name("bfl_biasLen")), h.name("bfl_radialLen")), h.lit("0.35"))},
				ir.Assign{Target: h.name("bfl_rz"), Op: "+", Value: h.bin("*", h.bin("*", h.bin("/", fDirZ, h.name("bfl_biasLen")), h.name("bfl_radialLen")), h.lit("0.35"))},
			},
		},
		h.letS("bfl_rdist", h.call("max",
			h.call("sqrt", h.bin("+", h.bin("+",
				h.bin("*", h.name("bfl_rx"), h.name("bfl_rx")),
				h.bin("*", h.name("bfl_ry"), h.name("bfl_ry"))),
				h.bin("*", h.name("bfl_rz"), h.name("bfl_rz")))),
			h.lit("0.001"))),
		makeDV(
			h.bin("*", h.bin("*", h.bin("/", h.name("bfl_rx"), h.name("bfl_rdist")), fStr), dtExpr),
			h.bin("*", h.bin("*", h.bin("/", h.name("bfl_ry"), h.name("bfl_rdist")), fStr), dtExpr),
			h.bin("*", h.bin("*", h.bin("/", h.name("bfl_rz"), h.name("bfl_rdist")), fStr), dtExpr),
		),
	}
	radialBody = append(radialBody, applyDV...)

	// Force dispatch: if chains on fKind (u32). We compare with u32 literals.
	// kind 0=gravity, 1=wind, 2=turbulence, 3=orbit, 4=drag, 5=radial.
	forceDispatch := []ir.Stmt{
		h.letS("bfl_f", ir.Index{E: forcesBinding, Idx: h.name("bfl_fi")}),
		ir.If{Cond: h.bin("==", fKind, h.lit("0u")), Then: gravityBody},
		ir.If{Cond: h.bin("==", fKind, h.lit("1u")), Then: windBody},
		ir.If{Cond: h.bin("==", fKind, h.lit("2u")), Then: turbBody},
		ir.If{Cond: h.bin("==", fKind, h.lit("3u")), Then: orbitBody},
		ir.If{Cond: h.bin("==", fKind, h.lit("4u")), Then: dragBody},
		ir.If{Cond: h.bin("==", fKind, h.lit("5u")), Then: radialBody},
	}

	// Loop: for (var bfl_fi = 0u; bfl_fi < params.forceCount; bfl_fi++)
	return []ir.Stmt{
		ir.For{
			Init: ir.Var{Name: "bfl_fi", Type: ir.U32, Init: h.lit("0u")},
			Cond: h.bin("<", h.name("bfl_fi"), forceCountExpr),
			Post: ir.Assign{Target: h.name("bfl_fi"), Op: "+", Value: h.lit("1u")},
			Body: forceDispatch,
		},
	}
}
