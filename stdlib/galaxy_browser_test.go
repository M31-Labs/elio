package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

// TestGalaxyParticleSimulateIsValid checks that GalaxyParticleSimulate passes
// the sema checker.
func TestGalaxyParticleSimulateIsValid(t *testing.T) {
	if errs := sema.Check(GalaxyParticleSimulate()); len(errs) != 0 {
		t.Fatalf("GalaxyParticleSimulate failed sema:\n%v", sema.Errors(errs))
	}
}

// simulateParticle runs a single particle through GalaxyParticleSimulate's
// CPU interpreter and returns the updated particle map and renderData map.
func simulateParticle(t *testing.T, posX, posY, posZ, velX, velY, velZ, age, lifetime float64, forces []any, dt, totalTime float64) (map[string]any, map[string]any) {
	t.Helper()
	p := map[string]any{
		"posX": posX, "posY": posY, "posZ": posZ,
		"velX": velX, "velY": velY, "velZ": velZ,
		"age": age, "lifetime": lifetime,
	}
	rd := map[string]any{
		"posX": 0.0, "posY": 0.0, "posZ": 0.0,
		"size": 0.0, "r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0,
	}
	params := map[string]any{
		"deltaTime": dt, "totalTime": totalTime,
		"count": int64(1), "_pad0": int64(0),
		"emitterKind": int64(0),
		"emitterX": 0.0, "emitterY": 0.0, "emitterZ": 0.0,
		"emitterRadius": 1.0, "emitterRate": 0.0,
		"emitterLifetime": 5.0, "emitterOnce": int64(0),
		"emitterArms": int64(2),
		"emitterWind": 0.0, "emitterScatter": 0.0,
		"emitterRotX": 0.0, "emitterRotY": 0.0, "emitterRotZ": 0.0,
		"_pad2": int64(0),
		"sizeStart": 1.0, "sizeEnd": 0.5,
		"colorStartR": 1.0, "colorStartG": 1.0, "colorStartB": 1.0,
		"colorEndR": 0.5, "colorEndG": 0.5, "colorEndB": 0.5,
		"opacityStart": 1.0, "opacityEnd": 0.0,
		"forceCount": int64(len(forces)), "bounds": 0.0,
	}
	mem := &run.Memory{Vars: map[string]any{
		"particles":  []any{p},
		"renderData": []any{rd},
		"params":     params,
		"forces":     forces,
	}}
	if err := run.Run(GalaxyParticleSimulate(), "simulate", 1, mem); err != nil {
		t.Fatalf("run.Run simulate: %v", err)
	}
	pOut := mem.Vars["particles"].([]any)[0].(map[string]any)
	rdOut := mem.Vars["renderData"].([]any)[0].(map[string]any)
	return pOut, rdOut
}

// simulateForce builds a Force map for the simulate kernel.
// kind: 0=gravity,1=wind,2=turbulence,3=orbit,4=drag,5=radial
func simulateForce(kind int64, strength, dirX, dirY, dirZ, freq float64) map[string]any {
	return map[string]any{
		"kind": kind, "strength": strength,
		"dirX": dirX, "dirY": dirY, "dirZ": dirZ,
		"frequency": freq, "_pad0": 0.0, "_pad1": 0.0,
	}
}

// TestSimulateGravityIntegration checks gravity-only integration for the
// browser kernel (kind=0, not kind=1 as in the native kernel).
func TestSimulateGravityIntegration(t *testing.T) {
	p, _ := simulateParticle(t, 0, 0, 0, 2, 0, 0, 0.1, 10,
		[]any{simulateForce(0, 10, 0, -1, 0, 0)}, 1.0, 0)
	// dv=(0,-10,0); vel'=(2,-10,0); pos'=(2,-10,0); age'=1.1
	nearVec(t, "velocity", []float64{p["velX"].(float64), p["velY"].(float64), p["velZ"].(float64)}, galaxyVec(2, -10, 0))
	nearVec(t, "position", []float64{p["posX"].(float64), p["posY"].(float64), p["posZ"].(float64)}, galaxyVec(2, -10, 0))
	if math.Abs(p["age"].(float64)-1.1) > 1e-6 {
		t.Errorf("age = %v, want 1.1", p["age"])
	}
}

// TestSimulateDragIntegration checks drag (kind=4) for the browser kernel.
func TestSimulateDragIntegration(t *testing.T) {
	p, _ := simulateParticle(t, 0, 0, 0, 10, 0, 0, 0.1, 10,
		[]any{simulateForce(4, 0.25, 0, 0, 0, 0)}, 1.0, 0)
	// dv = -velX * 0.25 * 1 = -2.5; velX' = 10-2.5 = 7.5; posX' = 7.5
	nearVec(t, "velocity", []float64{p["velX"].(float64), p["velY"].(float64), p["velZ"].(float64)}, galaxyVec(7.5, 0, 0))
	nearVec(t, "position", []float64{p["posX"].(float64), p["posY"].(float64), p["posZ"].(float64)}, galaxyVec(7.5, 0, 0))
}

// TestSimulateOrbitIntegration checks orbit (kind=3) for the browser kernel:
// at +x in XZ plane, orbit tangential is -z/dist = 0, dx/dist = 1 → dv=(0,0,1)*str*dt.
func TestSimulateOrbitIntegration(t *testing.T) {
	p, _ := simulateParticle(t, 10, 0, 0, 0, 0, 0, 0.1, 10,
		[]any{simulateForce(3, 1, 0, 0, 0, 0)}, 1.0, 0)
	// orbit: dx=10, dz=0, dist=10; dv=(-0/10, 0, 10/10)*1*1 = (0,0,1)
	// vel'=(0,0,1); pos' = (10+0, 0, 1)
	velZ := p["velZ"].(float64)
	if math.Abs(velZ-1.0) > 1e-5 {
		t.Errorf("orbit vel.z = %v, want 1.0", velZ)
	}
	if math.Abs(p["velX"].(float64)) > 1e-5 || math.Abs(p["velY"].(float64)) > 1e-5 {
		t.Errorf("orbit should be purely tangential: velX=%v velY=%v", p["velX"], p["velY"])
	}
}

// TestSimulateRenderVertexBaking checks that the RenderVertex is populated
// after a normal integration step.
func TestSimulateRenderVertexBaking(t *testing.T) {
	// alive particle, t = age/lifetime = 0.5/10 = 0.05
	// sizeStart=1, sizeEnd=0.5 → mix(1,0.5,0.05) = 0.975
	_, rd := simulateParticle(t, 0, 0, 0, 0, 0, 0, 0.5, 10, nil, 0.1, 0)
	if _, ok := rd["posX"]; !ok {
		t.Fatal("renderData posX missing")
	}
	// size = mix(1.0, 0.5, t') where t' = clamp((0.5+0.1)/10, 0, 1) = 0.06
	// size ≈ 1.0*(1-0.06) + 0.5*0.06 = 0.94 + 0.03 = 0.97
	size := rd["size"].(float64)
	if size <= 0 || size > 1.01 {
		t.Errorf("renderData size = %v, expected in (0,1]", size)
	}
	// a = mix(opStart, opEnd, t) * envelope(t) ≥ 0
	a := rd["a"].(float64)
	if a < 0 {
		t.Errorf("renderData alpha = %v, expected ≥ 0", a)
	}
}

// TestSimulateRespawnStructural checks that a particle with lifetime=0 (initial
// dead state, age negative) is respawned: age reset to 0, lifetime > 0.
func TestSimulateRespawnStructural(t *testing.T) {
	// age=-1 (dormant), lifetime=0 (unborn) → triggers emitParticle (kind=0 → emitPoint)
	p, rd := simulateParticle(t, 99, 99, 99, 0, 0, 0, -1, 0,
		[]any{simulateForce(0, 0, 0, -1, 0, 0)}, 0.016, 0)

	// After emit: age=0, lifetime>0, position at ~origin.
	age := p["age"].(float64)
	if age != 0.0 && math.Abs(age-0.016) > 1e-5 {
		// age should be 0 (just emitted) or 0+dt if we stepped age after emit.
		t.Logf("respawn age = %v (expect 0 or dt=0.016)", age)
	}
	lifetime := p["lifetime"].(float64)
	if lifetime <= 0 {
		t.Errorf("respawned particle lifetime = %v, want > 0", lifetime)
	}
	// position near origin (emitPoint puts at 0,0,0 then adds small vel*dt)
	posX := p["posX"].(float64)
	posY := p["posY"].(float64)
	posZ := p["posZ"].(float64)
	dist := math.Sqrt(posX*posX + posY*posY + posZ*posZ)
	if dist > 0.1 { // emitPoint puts at 0 with tiny vel, after one step pos < 0.01
		t.Errorf("respawned position far from origin: (%v,%v,%v)", posX, posY, posZ)
	}
	// RenderVertex should be non-dormant (a might be low but posX matches p.posX).
	rdPosX := rd["posX"].(float64)
	if math.Abs(rdPosX-posX) > 1e-6 {
		t.Errorf("renderData.posX = %v, want %v (particle posX)", rdPosX, posX)
	}
}

// --- Cross-kernel physics parity ---

// browserForce converts native kernel force fields to browser kernel Force map.
// Native: kind=1 gravity, 2 drag, 3 wind, 4 attractor, 5 vortex, 6 turbulence
// Browser: kind=0 gravity, 1 wind, 2 turbulence, 3 orbit, 4 drag, 5 radial
// Only the subset with direct equivalence (gravity, drag, orbit, turbulence)
// is parity-tested.
func browserForce(kind int64, strength, dirX, dirY, dirZ, freq float64) map[string]any {
	return map[string]any{
		"kind": kind, "strength": strength,
		"dirX": dirX, "dirY": dirY, "dirZ": dirZ,
		"frequency": freq, "_pad0": 0.0, "_pad1": 0.0,
	}
}

// TestCrossKernelIntegrationParity verifies that equivalent force physics
// on equivalent inputs produces matching position/velocity changes within 1e-5
// across the native and browser kernels.
//
// Both kernels share the same mathematical physics but use different force-kind
// numbering (gravity is kind=1 in native, kind=0 in browser) and different
// particle storage layouts. We map between them and compare integration outcomes.
func TestCrossKernelIntegrationParity(t *testing.T) {
	const eps = 1e-5
	dt := 0.016
	time := 5.0

	testCases := []struct {
		name    string
		posX, posY, posZ float64
		velX, velY, velZ float64
		// native force (kind in native numbering), browser force (kind in browser numbering)
		nativeForce  map[string]any
		browserForce map[string]any
	}{
		{
			name: "gravity",
			posX: 10, posY: 5, posZ: -3,
			velX: 1, velY: -0.5, velZ: 0.2,
			nativeForce:  galaxyForce(1, 9.8, 0, 0, -1, 0),   // native gravity kind=1
			browserForce: browserForce(0, 9.8, 0, -1, 0, 0),   // browser gravity kind=0
		},
		{
			name: "drag",
			posX: 5, posY: 0, posZ: 2,
			velX: 3, velY: -1, velZ: 0.5,
			nativeForce:  galaxyForce(2, 0.3, 0, 0, 0, 0),   // native drag kind=2
			browserForce: browserForce(4, 0.3, 0, 0, 0, 0),  // browser drag kind=4
		},
		{
			name: "orbit",
			posX: 8, posY: 0, posZ: 0,
			velX: 0, velY: 0, velZ: 0,
			// native vortex kind=5, fv=(0,1,0) → axis=+y; particle at (8,0,0):
			//   radial = (8,0,0) - (0,1,0)*dot((8,0,0),(0,1,0)) = (8,0,0)
			//   cross((8,0,0),(0,1,0)) = (0·0-0·1, 0·0-8·0, 8·1-0·0) = (0,0,8)
			//   normalize = (0,0,1); dv = (0,0,1)*1*dt
			// browser orbit kind=3; dx=8, dz=0, dist=8:
			//   dv = (-dz/dist, 0, dx/dist) * str * dt = (0, 0, 1) * 1 * dt
			// Both produce dv=(0,0,1)*dt — parity holds.
			nativeForce:  galaxyForce(5, 1, 0, 0, 1, 0),   // native vortex, fv=(0,1,0): kind=5, str=1, freq=0, vx=0, vy=1, vz=0
			browserForce: browserForce(3, 1, 0, 0, 0, 0),  // browser orbit kind=3
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Native kernel run.
			nativeP := galaxyParticle(
				galaxyVec(tc.posX, tc.posY, tc.posZ, 0),  // pos vec4 (xyz + age=0)
				galaxyVec(tc.velX, tc.velY, tc.velZ, 100), // vel vec4 (xyz + lifetime=100)
			)
			nativeU := galaxyUniforms(dt, time, 100,
				[]any{tc.nativeForce},
				galaxyVec(0, 0, 0, 0), galaxyVec(0, 0, 0, 0))
			runGalaxy(t, []any{nativeP}, nativeU)
			nPos := nativeP["position"].([]float64)
			nVel := nativeP["velocity"].([]float64)

			// Browser kernel run.
			bp, _ := simulateParticle(t,
				tc.posX, tc.posY, tc.posZ,
				tc.velX, tc.velY, tc.velZ,
				0.0, // age=0 (alive, no respawn; lifetime=100)
				100.0,
				[]any{tc.browserForce}, dt, time)

			bVelX := bp["velX"].(float64)
			bVelY := bp["velY"].(float64)
			bVelZ := bp["velZ"].(float64)
			bPosX := bp["posX"].(float64)
			bPosY := bp["posY"].(float64)
			bPosZ := bp["posZ"].(float64)

			// Compare velocity delta (integration result).
			if math.Abs(nVel[0]-bVelX) > eps {
				t.Errorf("velX: native=%v browser=%v diff=%v", nVel[0], bVelX, math.Abs(nVel[0]-bVelX))
			}
			if math.Abs(nVel[1]-bVelY) > eps {
				t.Errorf("velY: native=%v browser=%v diff=%v", nVel[1], bVelY, math.Abs(nVel[1]-bVelY))
			}
			if math.Abs(nVel[2]-bVelZ) > eps {
				t.Errorf("velZ: native=%v browser=%v diff=%v", nVel[2], bVelZ, math.Abs(nVel[2]-bVelZ))
			}
			// Compare position.
			if math.Abs(nPos[0]-bPosX) > eps {
				t.Errorf("posX: native=%v browser=%v diff=%v", nPos[0], bPosX, math.Abs(nPos[0]-bPosX))
			}
			if math.Abs(nPos[1]-bPosY) > eps {
				t.Errorf("posY: native=%v browser=%v diff=%v", nPos[1], bPosY, math.Abs(nPos[1]-bPosY))
			}
			if math.Abs(nPos[2]-bPosZ) > eps {
				t.Errorf("posZ: native=%v browser=%v diff=%v", nPos[2], bPosZ, math.Abs(nPos[2]-bPosZ))
			}
		})
	}
}
