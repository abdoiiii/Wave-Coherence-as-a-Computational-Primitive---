# Wave Mechanics: Structural Geometry Catalog

## Purpose

Stripped version of the full geometric relationship catalog. Everything astrological has been removed. What remains is pure mathematics: how to divide a circle, measure relationships by angle, handle non-geometric pairings, traverse directed cycles, and score relevance by context.

This is the specification for what the wave engine actually computes.

---

## 1. Phase Encoding

An attribute value is encoded as a point on the unit circle:

```
encode(value) → Complex64 { re: cos(θ), im: sin(θ) }
where θ = hash(value) * 2π / bucket_count
```

A **wave packet** bundles an entity's encoded attributes:

```
WavePacket {
    entity_id: u64,
    phases: HashMap<AttributeKey, Complex64>,
}
```

A **resonance field** is a collection of wave packets:

```
ResonanceField {
    packets: Vec<WavePacket>,
}
```

**Coherence** between two complex values measures their alignment:

```
coherence(a, b) = cos(angle(a) - angle(b))
  = 1.0  when identical (0° apart)
  = 0.0  when orthogonal (90° apart)
  = -1.0 when opposite (180° apart)
```

This is the dot product of two unit vectors. Nothing exotic.

---

## 2. Resolution Tiers (Bucket Counts)

The number of buckets determines encoding precision. More buckets = finer distinctions but larger scan space.

| Tier | Bucket count | Segment size | Use case |
|------|-------------|-------------|----------|
| 0 | 2 | 180.00° | Binary classification |
| 1 | 10–13 | 27–36° | Broad grouping |
| 2 | 20–27 | 13–18° | Category-level |
| 3 | 36–60 | 6–10° | Subcategory |
| 4 | 84–144 | 2.5–4.3° | Fine-grained |
| 5 | 192–360 | 1.0–1.9° | High precision |
| 6 | 480–720 | 0.5–0.75° | Maximum practical |

**Query planner rule:** Choose the tier where `expected_matches / bucket_count` is small enough to beat linear scan but large enough that the index lookup is worthwhile. Same heuristic as B-tree vs sequential scan in Postgres.

---

## 3. Harmonic Relationships (Angle-Based)

Dividing 360° by integer n produces the nth harmonic angle. Entities separated by that angle are "nth-harmonic related."

| Harmonic | Angle | Tolerance (orb) | Strength | Operation |
|----------|-------|----------------|----------|-----------|
| 1st | 0° | ±8° | 1.00 | Exact match |
| 2nd | 180° | ±8° | 0.90 | Complement |
| 3rd | 120° | ±8° | 0.85 | Harmonic family |
| 4th | 90° | ±7° | 0.80 | Tension / constraint |
| 5th | 72° | ±2° | 0.30 | Creative link |
| 6th | 60° | ±6° | 0.60 | Weak affinity |
| 8th | 45° | ±2° | 0.40 | Mild friction |
| 12th | 30° | ±2° | 0.30 | Adjacent / minimal |

**Mathematical basis:**

```
harmonic_coherence(a, b, n) = cos(n * (angle(a) - angle(b)))
```

When n=1, this is standard coherence. When n=3, it detects 120° relationships. The SAME coherence function with a frequency multiplier handles ALL harmonic relationships. No separate code paths needed.

**Tolerance (orb) as fuzzy matching:**

```
match_strength(a, b, target_angle, orb) =
    let delta = abs(angular_distance(a, b) - target_angle)
    if delta > orb: 0.0
    else: cos(delta * π / (2 * orb))  // smooth falloff from 1.0 at exact to 0.0 at edge
```

**Properties:** All symmetric. All purely geometric — computed from positions, no lookup tables needed. Lower harmonics = stronger base strength (this is Fourier energy distribution — fundamental frequency carries most energy).

---

## 4. Structural Pairings (Non-Geometric)

Some relationships exist because of explicit pairing rules, NOT angular distance. These require lookup tables.

### 4.1 Pair Tables

A pair table is a set of (position_A, position_B) tuples with a relationship label.

**Example: 6-pair compatibility table on a 12-position ring**

```
pairs = [
    (0, 1),   // positions 0 and 1 are compatible
    (2, 11),  // positions 2 and 11 are compatible
    (3, 10),  // etc.
    (4, 9),
    (5, 8),
    (6, 7),
]
```

These pairs have MIXED angular distances (30°, 90°, 150°, etc.). No single angle defines the relationship. The pairing exists because of TYPE properties of the positions (e.g., alternating binary attribute), not geometry.

### 4.2 Why This Matters for the Engine

The engine needs TWO query paths:

1. **Geometric:** "Find everything at 120° from X" → harmonic coherence calculation
2. **Structural:** "Find X's declared partner" → pair table lookup

A query can combine both: "Find entities that are harmonically related to X AND structurally paired with Y."

### 4.3 Multi-Body Structural Patterns

Some structural relationships involve 3+ entities:

**Triad groups:** 4 groups of 3 positions, each group separated by 120°
```
group_1 = [0, 4, 8]    // 120° apart
group_2 = [1, 5, 9]    // 120° apart
group_3 = [2, 6, 10]   // 120° apart
group_4 = [3, 7, 11]   // 120° apart
```

These happen to be geometric (120° = 3rd harmonic), but the GROUPING is structural — you're asserting that all three are in a named group, not just pairwise related.

**Destructive triangles:** Specific 3-position combinations that indicate conflict
```
pattern_1 = [2, 5, 8]   // mixed angles (90°/90°/180°)
pattern_2 = [1, 7, 10]  // mixed angles
```

**Self-referential:** Specific positions that conflict with themselves
```
self_conflict = [4, 6, 9, 11]
```

In DB terms: self-conflict detection = entity integrity check. "Does this entity's attributes create internal contradictions?"

### 4.4 Proximity Groups

Adjacent positions forming clusters:

```
cluster_1 = [2, 3, 4]     // consecutive positions
cluster_2 = [5, 6, 7]     // consecutive positions
cluster_3 = [8, 9, 10]    // consecutive positions
cluster_4 = [11, 0, 1]    // wraps around
```

In DB terms: range bucket. Entities within an angular arc are in the same functional cluster. Computed as:

```
same_cluster(a, b, span) = angular_distance(a, b) <= span
```

---

## 5. Directed Cycles

N nodes arranged in a ring with directed traversal at different step sizes.

### 5.1 The 5-Node Directed Cycle

5 equally spaced positions (0°, 72°, 144°, 216°, 288°).

Two interlocking directed graphs on the same 5 nodes:

```
Step +1 (generative):   0 → 1 → 2 → 3 → 4 → 0
Step +2 (destructive):  0 → 2 → 4 → 1 → 3 → 0
Step -1 (weakening):    0 → 4 → 3 → 2 → 1 → 0
Step -2 (controlling):  0 → 3 → 1 → 4 → 2 → 0
```

**Four relationship types from one structure:**

| Step | Relationship | DB Operation |
|------|-------------|-------------|
| +1 | A feeds B | Dependency: A is required for B |
| +2 | A overrides C | Conflict: A makes C invalid |
| -1 | A is drained by E | Cost: A consumes resources from E |
| -2 | A is constrained by D | Governance: D limits what A can do |

**Transitive queries:**
- "What does A enable?" → follow step +1 chain: A→B→C→D→E
- "What does removing A break?" → follow step -1 chain from A's dependents
- "What constrains A?" → follow step -2 to find A's governor

**Mathematical property:** Step +1 and step +2 generate two different Hamiltonian cycles on the same 5 nodes. They partition ALL possible directed edges into four relationship categories. Every pair of nodes has exactly one of four relationships. No ambiguity.

### 5.2 Dual-Cycle Compound Identity

Two independent cycles of different lengths (e.g., 10 and 12) produce a compound cycle of length LCM(10,12) = 60.

```
position_compound = (position_on_cycle_A, position_on_cycle_B)
```

Each entity has TWO independent phase angles. Relationships can be computed on either cycle independently or combined.

**DB implication:** Multi-dimensional phase encoding. An entity can participate in different relationship algebras simultaneously, like having both a "department" hierarchy and a "project" hierarchy.

---

## 6. Asymmetric Query Reach

Different entity types can detect relationships at different angles. Entity type determines which harmonics the entity "sees."

| Entity type | Angles visible |
|-------------|---------------|
| Default | 180° only |
| Type A | 90°, 180°, 210° |
| Type B | 120°, 180°, 240° |
| Type C | 60°, 180°, 270° |

**Properties:**

- ONE-DIRECTIONAL: Type A at position 0° sees entity at 90°, but entity at 90° does NOT see Type A back unless it's also Type A (or a type that sees 270° = 360°-90°).
- Query reach is parameterized, not hard-coded. Adding a new entity type with new visible angles requires only a config entry.
- Coherence calculation is the same math — you just filter which angles to check based on the querying entity's type.

```
typed_query(field, entity, entity_type) =
    for target_angle in visible_angles[entity_type]:
        results += scan_at_angle(field, entity.phase, target_angle, orb)
```

---

## 7. Context-Dependent Scoring

Entity relevance varies by domain context.

### 7.1 Domain Relevance Multiplier

```
relevance(entity_type, domain) → f64  // 0.0 to 1.0
```

Lookup table: (entity_type, domain) → multiplier.

| Level | Multiplier | Meaning |
|-------|-----------|---------|
| Home domain | 1.00 | Maximum relevance |
| Strong domain | 0.90 | Very relevant |
| Moderate domain | 0.70 | Somewhat relevant |
| Partial domain | 0.50 | Reduced relevance |
| Weak domain | 0.30 | Low relevance |
| Foreign domain | 0.15 | Minimal relevance |
| Opposing domain | 0.10 | Dysfunctional here |
| Worst domain | 0.05 | Severe mismatch |

Applied to query results: `final_score = coherence * relevance(type, current_domain)`

### 7.2 Binary Context Mode

System has a global mode (e.g., "mode_A" vs "mode_B") that swaps which entity types gain strength.

```
if mode == A:
    type_X.strength_bonus = +0.2
    type_Y.strength_bonus = -0.2
if mode == B:
    type_X.strength_bonus = -0.2
    type_Y.strength_bonus = +0.2
```

### 7.3 Mutual Reference Amplification

When entity A references B AND B references A, the relationship amplifies:

```
if A.references(B) AND B.references(A):
    coherence *= 1.5  // mutual
elif A.references(B) XOR B.references(A):
    coherence *= 1.2  // one-way
```

---

## 8. Compound Patterns

Multi-clause conditions that, when ALL met simultaneously, assert an emergent property.

```
Pattern {
    name: String,
    clauses: Vec<Clause>,
    result: EmergentProperty,
}

Clause {
    entity_selector: Selector,       // type, tag, phase range
    condition: Condition,            // min_refs, in_domain, related_to, etc.
}
```

**Evaluation:** Scan field. For each pattern, check if all clauses are satisfied simultaneously. If yes, assert the emergent property on matching entities.

**Reactive behavior:** When any entity's state changes, re-evaluate all patterns involving its type. Assert or retract emergent properties accordingly. This is a materialized view that stays consistent automatically.

---

## 9. Mathematical Operations Summary

Every operation the engine needs, with its computational signature:

| # | Operation | Input | Output | Complexity |
|---|-----------|-------|--------|-----------|
| 1 | Encode value | attribute value, bucket_count | Complex64 | O(1) |
| 2 | Coherence | two Complex64 | f64 [-1, 1] | O(1) |
| 3 | Harmonic coherence | two Complex64, harmonic n | f64 [-1, 1] | O(1) |
| 4 | Fuzzy match | two angles, target, orb | f64 [0, 1] | O(1) |
| 5 | Field scan (exact) | field, target phase | Vec<(entity_id, score)> | O(n) |
| 6 | Field scan (harmonic) | field, target phase, n | Vec<(entity_id, score)> | O(n) |
| 7 | Structural pair lookup | entity position, pair table | Option<entity_id> | O(1) |
| 8 | Directed cycle step | position, step, cycle_size | position | O(1) |
| 9 | Directed chain | position, step, depth | Vec<position> | O(depth) |
| 10 | Domain relevance | entity_type, domain | f64 | O(1) |
| 11 | Typed reach scan | field, entity, type_angles | Vec<(entity_id, score)> | O(n × angles) |
| 12 | Compound pattern | field, pattern_def | Vec<entity_id> | O(n × clauses) |
| 13 | Multi-field join | fields[], join condition | Vec<tuple> | O(n × m) |
| 14 | Resolution re-bucket | phase, source_tier, target_tier | phase | O(1) |
| 15 | Harmonic disambiguation | ambiguous candidates, max_n | unique entity_id | O(k × max_n) |

**Key observation:** Operations 1–6 are all variations of the same core math: complex number multiplication and angle comparison. The engine's inner loop is tiny. Everything else is configuration.

**Operation 15 note:** When a single-harmonic scan returns multiple candidates with identical coherence (bucket collision), higher harmonics disambiguate them. Each value has a unique *harmonic fingerprint* — the vector of coherence scores across harmonics 1, 2, ..., n. By the Fourier uniqueness theorem, no two distinct values produce the same fingerprint across all harmonics. Collision resolution scales by analysis depth (more harmonics), not storage (more buckets). The required harmonic is deterministic: `n_diverge = ⌈arccos(threshold) / Δθ⌉`, validated empirically with exact prediction-to-measurement agreement at angular differences of 2°, 1°, and 0.1°.

**Threshold scaling note:** For operations 5–6, the minimum coherence threshold for single-value precision depends on both bucket count B and harmonic number n. At harmonic n, the effective angular spacing is n × (2π/B), so the threshold floor is `cos(n × 2π / B)`. Higher harmonics amplify bucket spacing, requiring either tighter thresholds or more buckets for the same selectivity. The required bucket count for single-value precision at harmonic n with threshold t is: `B > n × 2π / arccos(t)`.

**Density scaling note:** For N objects on a B-bucket circle, exact match requires density < 100% (no two objects in the same bucket). Harmonic queries degrade earlier: triadic (n=3) detection at threshold 0.85 becomes noisy when minimum pairwise separation falls below ~10°. The resolution harmonic needed for the closest pair follows `max_n = ⌈arccos(t) / min_sep⌉`. Bucket collision probability follows the birthday problem: `P(collision) ≈ 1 - e^(-N²/2B)`.

**Self-indexing note:** Circular phase encoding is inherently self-indexing. Because the encoded value determines the angular position and the position determines the storage bucket, insertion simultaneously stores and indexes the entity. No separate index structure (B-tree, hash map) is required. Queries compute target bucket(s) and check only the relevant neighborhood. Exact query examines `⌈arccos(threshold) / (2π/B)⌉` buckets per side; harmonic query examines n regions each with `⌈arccos(threshold) / (n × 2π/B)⌉` buckets. Insert is O(1), queries are sub-linear.

**Multi-attribute torus note:** Multiple phase-encoded attributes compose into an N-torus (product of circles). Two attributes map to a 2-torus indexed by a B×B grid; three attributes map to a 3-torus indexed by a B×B×B grid. Compound queries check a rectangular neighborhood on the torus. Each dimension narrows independently, so selectivity improves multiplicatively: the combined selectivity approaches the product of per-dimension selectivities. This is the natural generalization of 1D bucket indexing to multi-column schemas.

**Dynamic mutation note:** The phase-indexed structure supports insert, remove, and update as local operations. Remove marks a tombstone and removes the entity's index from its bucket. Update is remove followed by re-insert at the new position. No global rebuild is required. Queries remain correct through arbitrary mutation sequences.

**Harmonic embedding note:** A single phase angle θ, probed across N harmonics, produces an N-dimensional vector:

```
v(θ) = [cos(θ), cos(2θ), cos(3θ), ..., cos(Nθ)]
```

This is a Fourier basis expansion — each component is a projection onto the nth harmonic. The dot product of two such vectors is the Dirichlet kernel: `dot(v(θ_a), v(θ_b)) = Σ cos(n × (θ_a - θ_b))` for n=1..N, which is the sum of all harmonic coherences. For K attributes, each probed across N harmonics, the combined vector has K×N dimensions:

```
v = [cos(θ₁), cos(2θ₁), ..., cos(Nθ₁), cos(θ₂), cos(2θ₂), ..., cos(Nθ₂), ..., cos(θ_K), ..., cos(Nθ_K)]
```

This is a structured embedding: each dimension has a defined meaning (nth harmonic of attribute k). The harmonic fingerprint validated in Test 11 is the 1-attribute case. The multi-attribute generalization produces embeddings of arbitrary dimensionality where the basis is given by construction, not learned. The dot product between two such embeddings captures all harmonic relationships across all attributes in a single operation.

---

## Summary

The engine computes exactly these things:

1. **Encode** values as angles on a circle
2. **Measure** alignment between angles (coherence)
3. **Scale** the frequency to detect harmonic relationships
4. **Look up** structural pairings from explicit tables
5. **Traverse** directed cycles for dependency/conflict chains
6. **Filter** by entity type for asymmetric reach
7. **Weight** by domain context for relevance scoring
8. **Match** compound patterns for emergent properties

Items 1–3 are pure wave mechanics (complex arithmetic).
Items 4–5 are graph primitives (lookup + traversal).
Items 6–8 are scoring/filtering layers.

The hypothesis: items 1–3 can replace traditional WHERE + JOIN for relationship-heavy queries with equal correctness and better performance on dense graphs. Items 4–8 extend the algebra beyond what pure geometry can express.

The test program validates items 1–3 first.
