package stdlib

import (
	"math"
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func TestGalaxyParticleUpdateIsValid(t *testing.T) {
	if errs := sema.Check(GalaxyParticleUpdate()); len(errs) != 0 {
		t.Fatalf("GalaxyParticleUpdate failed sema:\n%v", sema.Errors(errs))
	}
}

func galaxyVec(xs ...float64) []float64 { return append([]float64(nil), xs...) }

func galaxyParticle(pos, vel []float64) map[string]any {
	return map[string]any{"position": pos, "velocity": vel}
}

func galaxyForce(kind, strength, freq, vx, vy, vz float64) map[string]any {
	return map[string]any{"cfg": galaxyVec(kind, strength, freq, 0), "vector": galaxyVec(vx, vy, vz, 0)}
}

func galaxyUniforms(dt, time, lifetime float64, forces []any, emitter, initSpeed []float64) map[string]any {
	return map[string]any{
		"dt": dt, "time": time, "lifetime": lifetime, "forceCount": float64(len(forces)),
		"emitterPos": emitter, "initialSpeed": initSpeed, "forces": forces,
	}
}

func runGalaxy(t *testing.T, particles []any, u map[string]any) {
	t.Helper()
	mem := &run.Memory{Vars: map[string]any{"u": u, "particles": particles}}
	if err := run.Run(GalaxyParticleUpdate(), "update", len(particles), mem); err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func nearVec(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Errorf("%s[%d] = %v, want %v (got %v)", label, i, got[i], want[i], got)
		}
	}
}

// TestGalaxyGravityIntegration cross-checks the Elio kernel against GoSX's
// particleUpdateWGSL integration: v += accel*dt, p += v*dt, age += dt.
func TestGalaxyGravityIntegration(t *testing.T) {
	p := galaxyParticle(galaxyVec(0, 0, 0, 0), galaxyVec(2, 0, 0, 10))
	u := galaxyUniforms(1, 0, 10,
		[]any{galaxyForce(1, 10, 0, 0, -1, 0)}, // gravity strength 10, down
		galaxyVec(0, 0, 0, 0), galaxyVec(0, 0, 0, 0))
	runGalaxy(t, []any{p}, u)
	nearVec(t, "position", p["position"].([]float64), galaxyVec(2, -10, 0, 1))
	nearVec(t, "velocity", p["velocity"].([]float64), galaxyVec(2, -10, 0, 10))
}

// TestGalaxyDrag checks the multiplicative drag factor: v *= clamp(1-drag*dt).
func TestGalaxyDrag(t *testing.T) {
	p := galaxyParticle(galaxyVec(0, 0, 0, 0), galaxyVec(10, 0, 0, 10))
	u := galaxyUniforms(1, 0, 10,
		[]any{galaxyForce(2, 0.25, 0, 0, 0, 0)}, // drag 0.25
		galaxyVec(0, 0, 0, 0), galaxyVec(0, 0, 0, 0))
	runGalaxy(t, []any{p}, u)
	// dragFactor = 1 - 0.25*1 = 0.75; newVel.x = 10*0.75 = 7.5; newPos.x = 7.5.
	nearVec(t, "velocity", p["velocity"].([]float64), galaxyVec(7.5, 0, 0, 10))
	nearVec(t, "position", p["position"].([]float64), galaxyVec(7.5, 0, 0, 1))
}

// TestGalaxyVortexOrbit checks the vortex/orbit force: at +x around the +y axis,
// the tangential acceleration is +z — the swirl that gives the galaxy its arms.
func TestGalaxyVortexOrbit(t *testing.T) {
	p := galaxyParticle(galaxyVec(10, 0, 0, 0), galaxyVec(0, 0, 0, 10))
	u := galaxyUniforms(1, 0, 10,
		[]any{galaxyForce(5, 1, 0, 0, 1, 0)}, // vortex strength 1, axis +y
		galaxyVec(0, 0, 0, 0), galaxyVec(0, 0, 0, 0))
	runGalaxy(t, []any{p}, u)
	vel := p["velocity"].([]float64)
	if math.Abs(vel[2]-1) > 1e-6 {
		t.Errorf("vortex tangential velocity.z = %v, want 1", vel[2])
	}
	if math.Abs(vel[0]) > 1e-6 || math.Abs(vel[1]) > 1e-6 {
		t.Errorf("vortex should be purely tangential, got velocity %v", vel)
	}
}

// TestGalaxyRespawn checks the respawn path: an expired particle (life ≤ 0)
// reappears at the emitter (radius 0 ⇒ exact), age 0, lifetime reset.
func TestGalaxyRespawn(t *testing.T) {
	p := galaxyParticle(galaxyVec(99, 99, 99, 50), galaxyVec(1, 1, 1, 0)) // life 0 ⇒ respawn
	u := galaxyUniforms(1, 0.5, 12,
		[]any{galaxyForce(1, 10, 0, 0, -1, 0)},
		galaxyVec(5, 6, 7, 0), // emitter at (5,6,7), radius 0
		galaxyVec(3, 0, 0, 0)) // initial speed 3
	runGalaxy(t, []any{p}, u)
	pos := p["position"].([]float64)
	nearVec(t, "respawn position", pos[:3], galaxyVec(5, 6, 7))
	if math.Abs(pos[3]) > 1e-6 {
		t.Errorf("respawn age = %v, want 0", pos[3])
	}
	vel := p["velocity"].([]float64)
	if math.Abs(vel[3]-12) > 1e-6 {
		t.Errorf("respawn lifetime = %v, want 12", vel[3])
	}
	// velocity xyz = normalized dir * speed 3 ⇒ magnitude 3.
	mag := math.Sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2])
	if math.Abs(mag-3) > 1e-6 {
		t.Errorf("respawn speed = %v, want 3", mag)
	}
}
