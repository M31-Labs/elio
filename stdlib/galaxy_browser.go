package stdlib

import (
	"fmt"

	"m31labs.dev/elio/ir"
)

// GalaxyParticleSimulate is the browser-contract galaxy particle/star kernel:
// a bit-faithful Elio port of GoSX's SCENE_COMPUTE_PARTICLE_SOURCE from
// client/js/bootstrap-src/16b-scene-compute.js. It is the kernel the browser
// runtime compiles and dispatches instead of its built-in hardcoded source
// once the gosx payload channel delivers Elio-generated WGSL.
//
// Contract differences from GalaxyParticleUpdate (the native kernel):
//
//   - Particle layout: 8 scalar f32 fields (posX, posY, posZ, velX, velY,
//     velZ, age, lifetime) — same 32-byte stride as the native vec4 pair but
//     accessed as named scalars rather than swizzled vectors.
//
//   - RenderVertex output buffer (binding 1, storage rw): pre-bakes one
//     RenderVertex (posX/Y/Z, size, r/g/b/a) per particle per frame, consumed
//     directly by the browser renderer without an extra CPU readback pass.
//
//   - SimParams uniform (binding 2): 31 scalar fields covering emitter
//     geometry, material interpolation, force count, and bounds — laid out
//     sequentially with no implicit WGSL padding (all f32/u32 align 4).
//
//   - forces storage (binding 3, read-only): array<Force>, each Force being
//     kind/strength/dirX/dirY/dirZ/frequency plus 2 pad floats = 32 bytes.
//
//   - Randomness: the browser kernel uses a uint hash (hash/hash2 over u32
//     seeds derived from the particle index) rather than hash13 (the native
//     kernel's vec3→f32 Murmur-style hash). The two schemes produce different
//     pseudo-random streams and therefore different initial positions, but the
//     force-integration physics is identical once a particle is alive.
//
//   - Aging/respawn: richer state machine than the native kernel — dormant
//     phase (negative age counting toward birth), once-mode (particles that
//     expire stay finished forever), emitter kinds (point/sphere/disc/spiral),
//     and bounds culling are all reproduced.
//
//   - Entry name: "simulate" (vs "update" for the native kernel).
//
//   - Workgroup size: 64 (same as native).
//
// All existing GalaxyParticleUpdate tests pass unmodified; this function adds
// the browser-flavored second kernel alongside the native one.
func GalaxyParticleSimulate() *ir.Module {
	h := newExprHelpers()

	// Short aliases used throughout.
	name := h.name
	lit := h.lit
	mem := h.mem
	call := h.call
	bin := h.bin
	set := h.set
	addTo := h.addTo
	letS := h.letS

	mem2 := func(e ir.Expr, a, b string) ir.Expr { return ir.Member{E: ir.Member{E: e, Field: a}, Field: b} }
	_ = mem2

	params := func(f string) ir.Expr { return mem(name("params"), f) }
	pf := func(f string) ir.Expr { return mem(name("p"), f) }

	// vec3f constructor (browser uses vec3f not vec3<f32> — both are valid WGSL;
	// vec3f is the short alias; elio's WGSL emitter will emit vec3<f32> but naga
	// accepts both; for GLSL/Metal we emit float3/vec3 via the constructor path).
	v3f := func(x, y, z ir.Expr) ir.Expr { return call("vec3f", x, y, z) }
	_ = v3f

	// --- hash functions (browser kernel's u32-based randomness) ---
	// hash(seed u32) -> f32:
	//   s ^= s >> 16u; s *= 0x45d9f3bu; s ^= s >> 16u; s *= 0x45d9f3bu; s ^= s >> 16u
	//   return f32(s) / f32(0xffffffffu)
	//
	// We inline hash(seed) as a sequence of IR statements returning a named result.
	// Because Elio has no user-defined functions yet, every call site gets its own
	// inline expansion with a unique prefix to avoid name clashes.
	hashCnt := 0
	inlineHash := func(seedExpr ir.Expr) ([]ir.Stmt, ir.Expr) {
		hashCnt++
		p := fmt.Sprintf("hs%d_", hashCnt)
		ss := p + "s"
		stmts := []ir.Stmt{
			ir.Var{Name: ss, Type: ir.U32, Init: seedExpr},
			ir.Assign{Target: name(ss), Op: "^", Value: bin(">>", name(ss), lit("16u"))},
			ir.Assign{Target: name(ss), Op: "*", Value: lit("0x45d9f3bu")},
			ir.Assign{Target: name(ss), Op: "^", Value: bin(">>", name(ss), lit("16u"))},
			ir.Assign{Target: name(ss), Op: "*", Value: lit("0x45d9f3bu")},
			ir.Assign{Target: name(ss), Op: "^", Value: bin(">>", name(ss), lit("16u"))},
		}
		result := bin("/", call("f32", name(ss)), call("f32", lit("0xffffffffu")))
		return stmts, result
	}

	// hash2(a u32, b u32) -> f32 = hash(a * 1597334677u + b * 3812015801u)
	inlineHash2 := func(aExpr, bExpr ir.Expr) ([]ir.Stmt, ir.Expr) {
		seed := bin("+",
			bin("*", aExpr, lit("1597334677u")),
			bin("*", bExpr, lit("3812015801u")))
		return inlineHash(seed)
	}

	// --- particleLifetime helper (inlined, used by emitX and makeDormant) ---
	// particleLifetime(index, p):
	//   base = max(p.emitterLifetime, 0.001)
	//   return base * (0.64 + hash2(index, 44u) * 0.92)
	inlineParticleLifetime := func(indexExpr ir.Expr, prefix string) ([]ir.Stmt, ir.Expr) {
		hStmts, hExpr := inlineHash2(indexExpr, lit("44u"))
		stmts := []ir.Stmt{
			letS(prefix+"lt_base", call("max", params("emitterLifetime"), lit("0.001"))),
		}
		stmts = append(stmts, hStmts...)
		result := bin("*", name(prefix+"lt_base"), bin("+", lit("0.64"), bin("*", hExpr, lit("0.92"))))
		return stmts, result
	}

	// --- particleDelay helper (inlined for makeDormant) ---
	// particleDelay(index, p):
	//   base = max(p.emitterLifetime, 0.001)
	//   if emitterOnce != 0u: return pow(hash2(idx,47u),1.8) * max(0.08, base*0.12)
	//   window = base
	//   if emitterRate > 0.001: window = max(base*0.35, f32(count)/emitterRate)
	//   wave = floor(hash2(idx,46u)*5.0)/5.0
	//   jitter = pow(hash2(idx,47u), 2.2) * 0.18
	//   return min(window, (wave+jitter)*window)
	//
	// We inline this as a Var + if, writing result into a named variable.
	inlineParticleDelay := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		h46Stmts, h46 := inlineHash2(indexExpr, lit("46u"))
		h47aStmts, h47a := inlineHash2(indexExpr, lit("47u"))
		h47bStmts, h47b := inlineHash2(indexExpr, lit("47u"))

		stmts := []ir.Stmt{
			letS(prefix+"pd_base", call("max", params("emitterLifetime"), lit("0.001"))),
			ir.Var{Name: prefix + "pd_result", Init: lit("0.0")},
			ir.If{
				Cond: bin("!=", params("emitterOnce"), lit("0u")),
				Then: func() []ir.Stmt {
					s := h47aStmts
					s = append(s, set(name(prefix+"pd_result"),
						bin("*", call("pow", h47a, lit("1.8")), call("max", lit("0.08"), bin("*", name(prefix+"pd_base"), lit("0.12"))))))
					return s
				}(),
				Else: func() []ir.Stmt {
					s := []ir.Stmt{
						ir.Var{Name: prefix + "pd_window", Init: name(prefix + "pd_base")},
						ir.If{
							Cond: bin(">", params("emitterRate"), lit("0.001")),
							Then: []ir.Stmt{
								set(name(prefix+"pd_window"), call("max",
									bin("*", name(prefix+"pd_base"), lit("0.35")),
									bin("/", call("f32", params("count")), params("emitterRate")))),
							},
						},
					}
					s = append(s, h46Stmts...)
					s = append(s, letS(prefix+"pd_wave", bin("/", call("floor", bin("*", h46, lit("5.0"))), lit("5.0"))))
					s = append(s, h47bStmts...)
					s = append(s, letS(prefix+"pd_jitter", bin("*", call("pow", h47b, lit("2.2")), lit("0.18"))))
					s = append(s, set(name(prefix+"pd_result"),
						call("min", name(prefix+"pd_window"),
							bin("*", bin("+", name(prefix+"pd_wave"), name(prefix+"pd_jitter")), name(prefix+"pd_window")))))
					return s
				}(),
			},
		}
		return stmts
	}

	// --- particleEnvelope(t) = smoothstep(0,0.11,t) * (1 - smoothstep(0.66,1,t)) ---
	particleEnvelope := func(tExpr ir.Expr) ir.Expr {
		born := call("smoothstep", lit("0.0"), lit("0.11"), tExpr)
		spent := bin("-", lit("1.0"), call("smoothstep", lit("0.66"), lit("1.0"), tExpr))
		return bin("*", born, spent)
	}

	// --- writeDormant(i, p): zero out renderData slot ---
	// newZeroRenderVertex builds a zeroed RenderVertex map suitable for the CPU
	// interpreter (map[string]any). In the GPU path the Var init is a literal
	// zero map, but for the interpreter we need a real Go map. The GPU emitters
	// will see the field assignments below and emit the correct WGSL/GLSL/Metal.
	newZeroRenderVertex := func(varName string) []ir.Stmt {
		rv := name(varName)
		return []ir.Stmt{
			ir.Var{Name: varName, Type: ir.Named{Name: "RenderVertex"}, Init: ir.Lit{Text: "RenderVertex(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)"}},
			set(mem(rv, "posX"), lit("0.0")),
			set(mem(rv, "posY"), lit("0.0")),
			set(mem(rv, "posZ"), lit("0.0")),
			set(mem(rv, "size"), lit("0.0")),
			set(mem(rv, "r"), lit("1.0")),
			set(mem(rv, "g"), lit("1.0")),
			set(mem(rv, "b"), lit("1.0")),
			set(mem(rv, "a"), lit("0.0")),
		}
	}

	// We build writeDormant inline (not as a function) — write zero-alpha RenderVertex.
	writeDormantInline := func() []ir.Stmt {
		s := []ir.Stmt{set(ir.Index{E: name("particles"), Idx: name("sim_i")}, name("p"))}
		s = append(s, newZeroRenderVertex("wd_rv")...)
		s = append(s, set(ir.Index{E: name("renderData"), Idx: name("sim_i")}, name("wd_rv")))
		s = append(s, ir.Return{})
		return s
	}

	// --- makeFinished: age=-1, lifetime=-1, all zeros ---
	makeFinishedInto := func() []ir.Stmt {
		return []ir.Stmt{
			set(pf("posX"), lit("0.0")), set(pf("posY"), lit("0.0")), set(pf("posZ"), lit("0.0")),
			set(pf("velX"), lit("0.0")), set(pf("velY"), lit("0.0")), set(pf("velZ"), lit("0.0")),
			set(pf("age"), lit("-1.0")),
			set(pf("lifetime"), lit("-1.0")),
		}
	}

	// --- makeDormant(index, p): schedule delayed birth ---
	makeDormantInto := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		ltStmts, ltExpr := inlineParticleLifetime(indexExpr, prefix+"md_")
		pdStmts := inlineParticleDelay(indexExpr, prefix+"md_")
		stmts := []ir.Stmt{
			set(pf("posX"), lit("0.0")), set(pf("posY"), lit("0.0")), set(pf("posZ"), lit("0.0")),
			set(pf("velX"), lit("0.0")), set(pf("velY"), lit("0.0")), set(pf("velZ"), lit("0.0")),
		}
		stmts = append(stmts, ltStmts...)
		stmts = append(stmts, set(pf("lifetime"), ltExpr))
		stmts = append(stmts, pdStmts...)
		stmts = append(stmts, set(pf("age"), bin("*", lit("-1.0"), name(prefix+"md_pd_result"))))
		return stmts
	}

	// --- emitParticle helpers ---
	// emitPoint: near-origin with random small velocities
	emitPointInto := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		// hash2(index, 0u), hash2(index, 1u), hash2(index, 2u)
		h0s, h0 := inlineHash2(indexExpr, lit("0u"))
		h1s, h1 := inlineHash2(indexExpr, lit("1u"))
		h2s, h2 := inlineHash2(indexExpr, lit("2u"))
		ltStmts, ltExpr := inlineParticleLifetime(indexExpr, prefix+"ep_")

		s := []ir.Stmt{
			set(pf("posX"), lit("0.0")), set(pf("posY"), lit("0.0")), set(pf("posZ"), lit("0.0")),
		}
		s = append(s, h0s...)
		s = append(s, set(pf("velX"), bin("*", bin("-", h0, lit("0.5")), lit("0.1"))))
		s = append(s, h1s...)
		s = append(s, set(pf("velY"), bin("*", bin("-", h1, lit("0.5")), lit("0.1"))))
		s = append(s, h2s...)
		s = append(s, set(pf("velZ"), bin("*", bin("-", h2, lit("0.5")), lit("0.1"))))
		s = append(s, set(pf("age"), lit("0.0")))
		s = append(s, ltStmts...)
		s = append(s, set(pf("lifetime"), ltExpr))
		return s
	}

	// emitSphere: random point on/in sphere
	emitSphereInto := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		h10s, h10 := inlineHash2(indexExpr, lit("10u"))
		h11s, h11 := inlineHash2(indexExpr, lit("11u"))
		h12s, h12 := inlineHash2(indexExpr, lit("12u"))
		ltStmts, ltExpr := inlineParticleLifetime(indexExpr, prefix+"es_")

		s := h10s
		s = append(s, letS(prefix+"es_theta", bin("*", h10, lit("6.283185"))))
		s = append(s, h11s...)
		s = append(s, letS(prefix+"es_phi", call("acos", bin("-", bin("*", lit("2.0"), h11), lit("1.0")))))
		s = append(s, h12s...)
		s = append(s, letS(prefix+"es_r", bin("*", params("emitterRadius"), call("pow", h12, lit("0.333")))))
		s = append(s, set(pf("posX"), bin("*", bin("*", name(prefix+"es_r"), call("sin", name(prefix+"es_phi"))), call("cos", name(prefix+"es_theta")))))
		s = append(s, set(pf("posY"), bin("*", name(prefix+"es_r"), call("cos", name(prefix+"es_phi")))))
		s = append(s, set(pf("posZ"), bin("*", bin("*", name(prefix+"es_r"), call("sin", name(prefix+"es_phi"))), call("sin", name(prefix+"es_theta")))))
		s = append(s, set(pf("velX"), lit("0.0")), set(pf("velY"), lit("0.0")), set(pf("velZ"), lit("0.0")))
		s = append(s, set(pf("age"), lit("0.0")))
		s = append(s, ltStmts...)
		s = append(s, set(pf("lifetime"), ltExpr))
		return s
	}

	// emitDisc: random point on disc in XZ plane
	emitDiscInto := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		h20s, h20 := inlineHash2(indexExpr, lit("20u"))
		h21s, h21 := inlineHash2(indexExpr, lit("21u"))
		ltStmts, ltExpr := inlineParticleLifetime(indexExpr, prefix+"ed_")

		s := h20s
		s = append(s, letS(prefix+"ed_angle", bin("*", h20, lit("6.283185"))))
		s = append(s, h21s...)
		s = append(s, letS(prefix+"ed_r", bin("*", params("emitterRadius"), call("sqrt", h21))))
		s = append(s, set(pf("posX"), bin("*", name(prefix+"ed_r"), call("cos", name(prefix+"ed_angle")))))
		s = append(s, set(pf("posY"), lit("0.0")))
		s = append(s, set(pf("posZ"), bin("*", name(prefix+"ed_r"), call("sin", name(prefix+"ed_angle")))))
		s = append(s, set(pf("velX"), lit("0.0")), set(pf("velY"), lit("0.0")), set(pf("velZ"), lit("0.0")))
		s = append(s, set(pf("age"), lit("0.0")))
		s = append(s, ltStmts...)
		s = append(s, set(pf("lifetime"), ltExpr))
		return s
	}

	// emitSpiral: galaxy-arm spiral in XZ plane with optional rotation
	emitSpiralInto := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		h30s, h30 := inlineHash2(indexExpr, lit("30u"))
		h31s, h31 := inlineHash2(indexExpr, lit("31u"))
		h32s, h32 := inlineHash2(indexExpr, lit("32u"))
		h33s, h33 := inlineHash2(indexExpr, lit("33u"))
		ltStmts, ltExpr := inlineParticleLifetime(indexExpr, prefix+"esp_")

		s := h30s
		s = append(s, letS(prefix+"esp_radius", bin("*", h30, params("emitterRadius"))))
		// arm = index % emitterArms (both u32)
		s = append(s, letS(prefix+"esp_arm", bin("%", indexExpr, params("emitterArms"))))
		// armAngle = f32(arm) * pi / f32(max(emitterArms/2u, 1u))
		s = append(s, letS(prefix+"esp_armAngle",
			bin("*", call("f32", name(prefix+"esp_arm")),
				bin("/", lit("3.14159265"),
					call("f32", call("max", bin("/", params("emitterArms"), lit("2u")), lit("1u")))))))
		// spiralAngle = armAngle + (radius/emitterRadius)*emitterWind
		s = append(s, letS(prefix+"esp_spiralAngle",
			bin("+", name(prefix+"esp_armAngle"),
				bin("*", bin("/", name(prefix+"esp_radius"), params("emitterRadius")), params("emitterWind")))))
		s = append(s, h31s...)
		// scatter = (h31 - 0.5) * radius * emitterScatter
		s = append(s, letS(prefix+"esp_scatter", bin("*", bin("*", bin("-", h31, lit("0.5")), name(prefix+"esp_radius")), params("emitterScatter"))))
		// lx = cos(spiralAngle)*radius + scatter
		s = append(s, letS(prefix+"esp_lx", bin("+", bin("*", call("cos", name(prefix+"esp_spiralAngle")), name(prefix+"esp_radius")), name(prefix+"esp_scatter"))))
		s = append(s, h32s...)
		// ly = (h32 - 0.5) * emitterRadius * 0.05
		s = append(s, letS(prefix+"esp_ly", bin("*", bin("*", bin("-", h32, lit("0.5")), params("emitterRadius")), lit("0.05"))))
		s = append(s, h33s...)
		// lz = sin(spiralAngle)*radius + (h33-0.5)*radius*emitterScatter
		s = append(s, letS(prefix+"esp_lz", bin("+",
			bin("*", call("sin", name(prefix+"esp_spiralAngle")), name(prefix+"esp_radius")),
			bin("*", bin("*", bin("-", h33, lit("0.5")), name(prefix+"esp_radius")), params("emitterScatter")))))
		s = append(s, set(pf("posX"), name(prefix+"esp_lx")))
		s = append(s, set(pf("posY"), name(prefix+"esp_ly")))
		s = append(s, set(pf("posZ"), name(prefix+"esp_lz")))
		s = append(s, set(pf("velX"), lit("0.0")), set(pf("velY"), lit("0.0")), set(pf("velZ"), lit("0.0")))
		s = append(s, set(pf("age"), lit("0.0")))
		s = append(s, ltStmts...)
		s = append(s, set(pf("lifetime"), ltExpr))
		return s
	}

	// emitParticle dispatches on emitterKind matching the reference JS switch:
	// case 1=sphere, case 2=disc, case 3=spiral, default=point (covers 0 and any >=4).
	emitParticleInto := func(indexExpr ir.Expr, prefix string) []ir.Stmt {
		return []ir.Stmt{
			ir.If{
				Cond: bin("==", params("emitterKind"), lit("1u")),
				Then: emitSphereInto(indexExpr, prefix+"k1_"),
				Else: []ir.Stmt{ir.If{
					Cond: bin("==", params("emitterKind"), lit("2u")),
					Then: emitDiscInto(indexExpr, prefix+"k2_"),
					Else: []ir.Stmt{ir.If{
						Cond: bin("==", params("emitterKind"), lit("3u")),
						Then: emitSpiralInto(indexExpr, prefix+"k3_"),
						Else: emitPointInto(indexExpr, prefix+"k0_"),
					}},
				}},
			},
		}
	}

	_ = addTo
	_ = set

	// ---- kernel body ----
	// let i = id.x
	// if i >= params.count { return; }
	// var p = particles[i]
	//
	// // Aging / respawn state machine.
	// if p.age < 0.0 {
	//   if emitterOnce != 0u && p.lifetime < -0.5 { writeDormant; return; }
	//   if p.lifetime <= 0.0 { emitParticle }
	//   else { p.age += dt; if p.age < 0 { writeDormant; return; } emitParticle }
	// }
	// p.age += dt
	// if p.lifetime > 0 && p.age >= p.lifetime {
	//   if emitterOnce { makeFinished; writeDormant; return; }
	//   makeDormant; p.age += dt; if p.age < 0 { writeDormant; return; } emitParticle
	// }
	//
	// // Force integration.
	// [force loop]
	//
	// // Position integration.
	// p.posX += p.velX * dt; p.posY += ...; p.posZ += ...
	//
	// // Bounds check (emitterOnce only).
	// if emitterOnce && bounds > 0 && length(...) > bounds { makeFinished; writeDormant; return; }
	//
	// // Write state.
	// particles[i] = p;
	//
	// // Compute t; bake RenderVertex.
	// ...
	// renderData[i] = rv;

	simI := name("sim_i")

	// Force loop using shared physics helper.
	forceLoop := browserForceLoop(
		h,
		pf("posX"), pf("posY"), pf("posZ"),
		pf("velX"), pf("velY"), pf("velZ"),
		params("forceCount"),
		params("deltaTime"),
		params("totalTime"),
		name("forces"),
	)

	// Aging state machine block when p.age < 0:
	ageNegBlock := []ir.Stmt{
		ir.If{
			Cond: bin("&&", bin("!=", params("emitterOnce"), lit("0u")), bin("<", pf("lifetime"), lit("-0.5"))),
			Then: writeDormantInline(),
		},
		ir.If{
			Cond: bin("<=", pf("lifetime"), lit("0.0")),
			Then: emitParticleInto(simI, "en_"),
			Else: func() []ir.Stmt {
				s := []ir.Stmt{
					ir.Assign{Target: pf("age"), Op: "+", Value: params("deltaTime")},
					ir.If{
						Cond: bin("<", pf("age"), lit("0.0")),
						Then: writeDormantInline(),
					},
				}
				s = append(s, emitParticleInto(simI, "ea_")...)
				return s
			}(),
		},
	}

	// Life-expired block when p.age >= p.lifetime:
	lifeExpiredBlock := []ir.Stmt{
		ir.If{
			Cond: bin("!=", params("emitterOnce"), lit("0u")),
			Then: append(makeFinishedInto(), writeDormantInline()...),
		},
	}
	lifeExpiredBlock = append(lifeExpiredBlock, makeDormantInto(simI, "le_")...)
	lifeExpiredBlock = append(lifeExpiredBlock,
		ir.Assign{Target: pf("age"), Op: "+", Value: params("deltaTime")},
		ir.If{
			Cond: bin("<", pf("age"), lit("0.0")),
			Then: writeDormantInline(),
		},
	)
	lifeExpiredBlock = append(lifeExpiredBlock, emitParticleInto(simI, "lep_")...)

	// Bounds check block.
	boundsCheck := []ir.Stmt{
		ir.If{
			Cond: bin("&&",
				bin("&&", bin("!=", params("emitterOnce"), lit("0u")), bin(">", params("bounds"), lit("0.0"))),
				bin(">", call("length", v3f(pf("posX"), pf("posY"), pf("posZ"))), params("bounds"))),
			Then: append(makeFinishedInto(), writeDormantInline()...),
		},
	}

	// t = clamp(select(p.age / p.lifetime, 0.0, p.lifetime <= 0.0), 0.0, 1.0)
	tExpr := call("clamp",
		call("select",
			bin("/", pf("age"), pf("lifetime")), // falseVal: age/lifetime
			lit("0.0"),                           // trueVal: 0 (when cond is true)
			bin("<=", pf("lifetime"), lit("0.0"))), // cond: lifetime <= 0
		lit("0.0"), lit("1.0"))

	// RenderVertex baking.
	renderBake := []ir.Stmt{
		letS("sim_t", tExpr),
	}
	renderBake = append(renderBake, newZeroRenderVertex("sim_rv")...)
	renderBake = append(renderBake,
		set(mem(name("sim_rv"), "posX"), pf("posX")),
		set(mem(name("sim_rv"), "posY"), pf("posY")),
		set(mem(name("sim_rv"), "posZ"), pf("posZ")),
		set(mem(name("sim_rv"), "size"), call("mix", params("sizeStart"), params("sizeEnd"), name("sim_t"))),
		set(mem(name("sim_rv"), "r"), call("mix", params("colorStartR"), params("colorEndR"), name("sim_t"))),
		set(mem(name("sim_rv"), "g"), call("mix", params("colorStartG"), params("colorEndG"), name("sim_t"))),
		set(mem(name("sim_rv"), "b"), call("mix", params("colorStartB"), params("colorEndB"), name("sim_t"))),
		set(mem(name("sim_rv"), "a"), bin("*",
			call("mix", params("opacityStart"), params("opacityEnd"), name("sim_t")),
			particleEnvelope(name("sim_t")))),
		set(ir.Index{E: name("renderData"), Idx: simI}, name("sim_rv")),
	)

	body := []ir.Stmt{
		letS("sim_i", mem(name("gid"), "x")),
		ir.If{
			Cond: bin(">=", name("sim_i"), params("count")),
			Then: []ir.Stmt{ir.Return{}},
		},
		ir.Var{Name: "p", Type: ir.Named{Name: "Particle"}, Init: ir.Index{E: name("particles"), Idx: simI}},

		// Age < 0 → dormant/birth branch.
		ir.If{
			Cond: bin("<", pf("age"), lit("0.0")),
			Then: ageNegBlock,
		},

		// Advance age.
		ir.Assign{Target: pf("age"), Op: "+", Value: params("deltaTime")},

		// Life expired → respawn or finish.
		ir.If{
			Cond: bin("&&", bin(">", pf("lifetime"), lit("0.0")), bin(">=", pf("age"), pf("lifetime"))),
			Then: lifeExpiredBlock,
		},
	}
	body = append(body, forceLoop...)
	body = append(body,
		ir.Assign{Target: pf("posX"), Op: "+", Value: bin("*", pf("velX"), params("deltaTime"))},
		ir.Assign{Target: pf("posY"), Op: "+", Value: bin("*", pf("velY"), params("deltaTime"))},
		ir.Assign{Target: pf("posZ"), Op: "+", Value: bin("*", pf("velZ"), params("deltaTime"))},
	)
	body = append(body, boundsCheck...)
	body = append(body, set(ir.Index{E: name("particles"), Idx: simI}, name("p")))
	body = append(body, renderBake...)

	field := func(n string, t ir.Type) ir.Field { return ir.Field{Name: n, Type: t} }
	return &ir.Module{
		Structs: []ir.Struct{
			{Name: "Particle", Fields: []ir.Field{
				field("posX", ir.F32), field("posY", ir.F32), field("posZ", ir.F32),
				field("velX", ir.F32), field("velY", ir.F32), field("velZ", ir.F32),
				field("age", ir.F32), field("lifetime", ir.F32),
			}},
			{Name: "RenderVertex", Fields: []ir.Field{
				field("posX", ir.F32), field("posY", ir.F32), field("posZ", ir.F32),
				field("size", ir.F32),
				field("r", ir.F32), field("g", ir.F32), field("b", ir.F32), field("a", ir.F32),
			}},
			{Name: "SimParams", Fields: []ir.Field{
				field("deltaTime", ir.F32),
				field("totalTime", ir.F32),
				field("count", ir.U32),
				field("_pad0", ir.U32),
				field("emitterKind", ir.U32),
				field("emitterX", ir.F32), field("emitterY", ir.F32), field("emitterZ", ir.F32),
				field("emitterRadius", ir.F32),
				field("emitterRate", ir.F32),
				field("emitterLifetime", ir.F32),
				field("emitterOnce", ir.U32),
				field("emitterArms", ir.U32),
				field("emitterWind", ir.F32),
				field("emitterScatter", ir.F32),
				field("emitterRotX", ir.F32), field("emitterRotY", ir.F32), field("emitterRotZ", ir.F32),
				field("_pad2", ir.U32),
				field("sizeStart", ir.F32), field("sizeEnd", ir.F32),
				field("colorStartR", ir.F32), field("colorStartG", ir.F32), field("colorStartB", ir.F32),
				field("colorEndR", ir.F32), field("colorEndG", ir.F32), field("colorEndB", ir.F32),
				field("opacityStart", ir.F32), field("opacityEnd", ir.F32),
				field("forceCount", ir.U32),
				field("bounds", ir.F32),
			}},
			{Name: "Force", Fields: []ir.Field{
				field("kind", ir.U32),
				field("strength", ir.F32),
				field("dirX", ir.F32), field("dirY", ir.F32), field("dirZ", ir.F32),
				field("frequency", ir.F32),
				field("_pad0", ir.F32), field("_pad1", ir.F32),
			}},
		},
		Bindings: []ir.Binding{
			{Group: 0, Binding: 0, Space: ir.Storage, Access: ir.ReadWrite, Name: "particles", Type: ir.Array{Elem: ir.Named{Name: "Particle"}}},
			{Group: 0, Binding: 1, Space: ir.Storage, Access: ir.ReadWrite, Name: "renderData", Type: ir.Array{Elem: ir.Named{Name: "RenderVertex"}}},
			{Group: 0, Binding: 2, Space: ir.Uniform, Name: "params", Type: ir.Named{Name: "SimParams"}},
			{Group: 0, Binding: 3, Space: ir.Storage, Access: ir.Read, Name: "forces", Type: ir.Array{Elem: ir.Named{Name: "Force"}}},
		},
		Kernels: []ir.Kernel{{
			Name:          "simulate",
			WorkgroupSize: [3]int{64, 1, 1},
			Builtins:      []ir.Builtin{{Name: "gid", Builtin: "global_invocation_id", Type: ir.Vec{N: 3, Elem: ir.U32}}},
			Body:          body,
		}},
	}
}
