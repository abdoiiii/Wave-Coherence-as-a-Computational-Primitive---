# Wave Mechanics Test Program

## What This Is

A small Rust program (~300 lines) that validates the core math WITHOUT any database. No SurrealDB, no MCP, no server. Just:
- Encode values as complex numbers
- Compute coherence
- Check if harmonic detection actually works
- Run it, see if the math does what we claim

If this fails, there's no point building the database layer.

---

## Test Structure

```
wave-test/
├── Cargo.toml
├── src/
│   ├── main.rs          # Test runner
│   ├── wave.rs          # WavePacket, encode, coherence
│   ├── field.rs         # ResonanceField, scan operations
│   └── relationships.rs # Harmonic detection, pair tables, directed cycles
```

### Dependencies

```toml
[package]
name = "wave-test"
version = "0.1.0"
edition = "2021"

[dependencies]
# num-complex for Complex64 - or just use (f64, f64) tuple, zero deps
```

Could even do it with zero dependencies — Complex64 is just two f64s with cos/sin.

---

## Core Types

```rust
// wave.rs

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Phase(pub f64); // angle in radians [0, 2π)

impl Phase {
    pub fn from_value(value: u64, bucket_count: u32) -> Self {
        let angle = (value % bucket_count as u64) as f64 * 2.0 * PI / bucket_count as f64;
        Phase(angle)
    }

    pub fn from_degrees(degrees: f64) -> Self {
        Phase(degrees * PI / 180.0)
    }

    /// Core operation: coherence = cos(angle_a - angle_b)
    /// Returns: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
    pub fn coherence(&self, other: &Phase) -> f64 {
        (self.0 - other.0).cos()
    }

    /// Harmonic coherence: detects nth-harmonic relationships
    /// n=1: exact match. n=2: opposition. n=3: trine/120°. etc.
    pub fn harmonic_coherence(&self, other: &Phase, n: u32) -> f64 {
        (n as f64 * (self.0 - other.0)).cos()
    }

    /// Angular distance in degrees (always positive, 0-180)
    pub fn distance_degrees(&self, other: &Phase) -> f64 {
        let diff = (self.0 - other.0).abs() % (2.0 * PI);
        let d = if diff > PI { 2.0 * PI - diff } else { diff };
        d * 180.0 / PI
    }
}

#[derive(Clone, Debug)]
pub struct WavePacket {
    pub id: String,
    pub phases: Vec<(String, Phase)>, // (attribute_name, encoded_phase)
}

impl WavePacket {
    pub fn new(id: &str) -> Self {
        WavePacket { id: id.to_string(), phases: vec![] }
    }

    pub fn with_attr(mut self, name: &str, phase: Phase) -> Self {
        self.phases.push((name.to_string(), phase));
        self
    }

    /// Get phase for a specific attribute
    pub fn phase_for(&self, attr: &str) -> Option<&Phase> {
        self.phases.iter().find(|(n, _)| n == attr).map(|(_, p)| p)
    }
}
```

```rust
// field.rs

pub struct ResonanceField {
    pub packets: Vec<WavePacket>,
}

impl ResonanceField {
    pub fn new() -> Self {
        ResonanceField { packets: vec![] }
    }

    pub fn add(&mut self, packet: WavePacket) {
        self.packets.push(packet);
    }

    /// Exact match: find all packets where attr coherence > threshold
    pub fn query_exact(&self, attr: &str, target: &Phase, threshold: f64) -> Vec<(&WavePacket, f64)> {
        self.packets.iter()
            .filter_map(|p| {
                p.phase_for(attr).map(|phase| {
                    let c = phase.coherence(target);
                    if c >= threshold { Some((p, c)) } else { None }
                }).flatten()
            })
            .collect()
    }

    /// Harmonic scan: find all packets at nth harmonic of target
    pub fn query_harmonic(&self, attr: &str, target: &Phase, harmonic: u32, threshold: f64) -> Vec<(&WavePacket, f64)> {
        self.packets.iter()
            .filter_map(|p| {
                p.phase_for(attr).map(|phase| {
                    let c = phase.harmonic_coherence(target, harmonic);
                    if c >= threshold { Some((p, c)) } else { None }
                }).flatten()
            })
            .collect()
    }
}
```

---

## The Tests

### Test 1: "Guessing Game" — Can coherence find an exact match?

```
Setup:
  - Encode 12 values (0-11) onto a 12-bucket circle
  - Pick a target value (say, 7)
  - Scan field for coherence > 0.99

Expected:
  - Only value 7 returns coherence ≈ 1.0
  - All others return coherence < 0.99
  - Adjacent values (6, 8) have the highest non-match coherence

Pass condition: Correct value found, no false positives above threshold.
```

### Test 2: Harmonic Family Detection

```
Setup:
  - 12 entities at positions 0° through 330° (30° intervals)
  - Target: entity at 0°
  - Query: 3rd harmonic (120° relationships)

Expected:
  - Entities at 120° and 240° return harmonic_coherence ≈ 1.0
  - Entity at 0° itself returns harmonic_coherence = 1.0 (trivially)
  - All others return |harmonic_coherence| < threshold

Pass condition: Exactly the 120° group detected. No others.
```

### Test 3: Opposition Detection

```
Setup:
  - Same 12 entities
  - Target: entity at 0°
  - Query: 2nd harmonic (180° relationships)

Expected:
  - Entity at 180° returns harmonic_coherence ≈ 1.0
  - Entity at 0° itself returns 1.0

Pass condition: Opposition correctly identified.
```

### Test 4: Fuzzy Matching with Tolerance

```
Setup:
  - Entity at exactly 118° (close to 120° but not exact)
  - Entity at exactly 120°
  - Entity at exactly 125° (farther off)
  - Target: 0°
  - Query: 3rd harmonic with orb = 8°

Expected:
  - 120° entity: score ≈ 1.0 (exact)
  - 118° entity: score > 0.9 (within orb, high)
  - 125° entity: score > 0.7 (within orb, lower)
  - Entity at 130°: score below threshold (outside orb)

Pass condition: Scores degrade gracefully with distance from exact angle.
```

### Test 5: Multi-Attribute Coherence

```
Setup:
  - Entities with TWO attributes: "vendor" and "category"
  - Entity A: vendor=phase(30°), category=phase(120°)
  - Entity B: vendor=phase(30°), category=phase(240°)
  - Entity C: vendor=phase(30°), category=phase(120°)
  - Target: vendor=phase(30°), category=phase(120°)

Expected:
  - C matches perfectly on both attributes
  - A matches vendor but not category... wait, A also has category=120°
  - Let me redo: Entity B has vendor=30° (match) but category=240° (not match)
  - Combined score for C > combined score for B

Pass condition: Multi-attribute conjunction works as AND — both must cohere.
```

### Test 6: Directed Cycle Traversal

```
Setup:
  - 5 positions: 0, 1, 2, 3, 4
  - Generative step = +1 mod 5
  - Destructive step = +2 mod 5

Test:
  - From position 0, generative chain depth 3: [0 → 1 → 2 → 3]
  - From position 0, destructive chain depth 3: [0 → 2 → 4 → 1]
  - From position 3, generative chain depth 2: [3 → 4 → 0]

Pass condition: Traversal matches expected paths exactly.
```

### Test 7: Structural Pair Lookup

```
Setup:
  - Pair table: [(0,1), (2,11), (3,10), (4,9), (5,8), (6,7)]
  - These have MIXED angular distances on a 12-position ring

Test:
  - Position 0's structural partner = 1 (angular distance 30°)
  - Position 2's structural partner = 11 (angular distance 90°)
  - Position 3's structural partner = 10 (angular distance 150°)
  - Position 0 is NOT structurally paired with position 4 (even though 120° is a harmonic)

Pass condition: Structural pairs found correctly. Geometric relationships do NOT imply structural pairing.
```

### Test 8: The Critical Comparison — Wave vs Linear Scan

```
Setup:
  - Generate 1000 entities with random attributes
  - Pick a target entity
  - Find all entities matching a specific attribute value

Method A (linear scan with string comparison):
  - Loop through all 1000, compare attribute string
  - Record time

Method B (coherence scan):
  - Encode all attributes as phases
  - Scan field for coherence > 0.99 on target attribute
  - Record time

Expected:
  - Both methods return IDENTICAL results (correctness validation)
  - At 1000 entities, timing difference is negligible
  - This test is about correctness, not performance

Pass condition: ZERO difference in result sets between methods.
```

### Test 9: The Value Proposition — Harmonic Query vs JOIN

```
Setup:
  - 100 entities in 4 groups (25 each), separated by 90° on the circle
  - Entities within each group are at similar phases (clustered)
  - Pre-define: group at 0° and group at 120° are "related"

Task: Find all entities related to entity X at 3° (in the 0° group)

Method A (traditional):
  - Determine X's group membership (WHERE group = 'A')
  - Look up related groups (JOIN group_relations ON ...)
  - Return all members of related groups

Method B (harmonic scan):
  - Compute 3rd harmonic coherence of X against entire field
  - Return all entities with coherence > threshold

Expected:
  - Both return the ~25 entities in the 120° group
  - Method B also returns the ~25 in the 240° group (BOTH trine positions)
  - Method A requires a second JOIN to find the 240° group
  - Method B found both in a single pass

Pass condition: Harmonic scan finds relationships that require multiple joins in the traditional approach, in a single operation.
```

### Test 10: Entity Type-Dependent Reach

```
Setup:
  - 12 entities around the circle at 30° intervals
  - Entity at 0° has type "broad" (sees 60°, 180°, 270°)
  - Entity at 0° has type "narrow" (sees only 180°)

Test with "broad" type from position 0°:
  - Should find entities at 60°, 180°, 270°

Test with "narrow" type from position 0°:
  - Should find entity at 180° only

Pass condition: Same position, different type, different results.
```

---

## What Success Looks Like

If ALL tests pass:

1. **Encoding is sound** — values map to phases and back without collision (Test 1)
2. **Harmonic detection works** — single coherence function finds relationships at any angle (Tests 2, 3)
3. **Fuzzy matching degrades gracefully** — partial matches score proportionally (Test 4)
4. **Multi-attribute conjunction works** — AND logic via coherence multiplication (Test 5)
5. **Directed cycles traverse correctly** — dependency/conflict chains computable (Test 6)
6. **Structural pairs override geometry** — the engine needs both paths (Test 7)
7. **Results match traditional queries** — zero false positives or negatives (Test 8)
8. **Harmonic queries find what JOINs miss** — unique value demonstrated (Test 9)
9. **Type-dependent reach works** — asymmetric queries from config (Test 10)

If any test FAILS, we know EXACTLY which part of the math breaks and can either fix it or acknowledge the limitation before writing a single line of database code.

---

## What to Run This On

This is a `cargo run` program. No server, no network, no database. Just:

```bash
cargo new wave-test
cd wave-test
# paste the source files
cargo run
```

Output should be a simple pass/fail for each test with the actual numbers printed for inspection.
