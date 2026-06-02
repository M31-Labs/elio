package stdlib

import (
	"testing"

	"m31labs.dev/elio/run"
	"m31labs.dev/elio/sema"
)

func TestParticleUpdateIsValid(t *testing.T) {
	if errs := sema.Check(ParticleUpdate()); len(errs) != 0 {
		t.Fatalf("ParticleUpdate failed sema:\n%v", sema.Errors(errs))
	}
}

// TestParticleUpdateIntegratesAndRespawns runs the kernel over two particles —
// one alive (integrated under gravity) and one expired (respawned at the
// emitter) — and checks both paths, executing the per-particle update on the CPU
// fallback exactly as a GPU backend would.
func TestParticleUpdateIntegratesAndRespawns(t *testing.T) {
	particle := func(px, py, pz, age, vx, vy, vz, life float64) map[string]any {
		return map[string]any{
			"px": px, "py": py, "pz": pz, "age": age,
			"vx": vx, "vy": vy, "vz": vz, "life": life,
		}
	}
	// p0 is alive (age 0, life 10); p1 has expired (age 9.5 + dt 1 ≥ life 10).
	p0 := particle(0, 0, 0, 0, 1, 0, 0, 10)
	p1 := particle(7, 7, 7, 9.5, 4, 5, 6, 10)
	particles := []any{p0, p1}

	sim := map[string]any{
		"dt":  1.0,
		"gx":  0.0, "gy": -10.0, "gz": 0.0, // gravity
		"ex":  0.0, "ey": 100.0, "ez": 0.0, // emitter origin
		"ivx": 1.0, "ivy": 2.0, "ivz": 3.0, // respawn velocity
	}
	mem := &run.Memory{Vars: map[string]any{"sim": sim, "particles": particles}}

	if err := run.Run(ParticleUpdate(), "update", len(particles), mem); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// p0 integrated: v += gravity*dt → (1,-10,0); p += v*dt → (1,-10,0); age = 1.
	wantAlive := particle(1, -10, 0, 1, 1, -10, 0, 10)
	for _, f := range []string{"px", "py", "pz", "age", "vx", "vy", "vz", "life"} {
		if p0[f] != wantAlive[f] {
			t.Errorf("alive particle .%s = %v, want %v", f, p0[f], wantAlive[f])
		}
	}
	// p1 respawned: position = emitter, age = 0, velocity = respawn velocity.
	wantRespawn := particle(0, 100, 0, 0, 1, 2, 3, 10)
	for _, f := range []string{"px", "py", "pz", "age", "vx", "vy", "vz", "life"} {
		if p1[f] != wantRespawn[f] {
			t.Errorf("respawned particle .%s = %v, want %v", f, p1[f], wantRespawn[f])
		}
	}
}
