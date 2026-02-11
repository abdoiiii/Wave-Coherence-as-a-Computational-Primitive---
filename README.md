[![DOI](https://zenodo.org/badge/1153530777.svg)](https://doi.org/10.5281/zenodo.18607190)

# Wave Coherence as a Computational Primitive

## What This Is

A validated mathematical framework that uses phase encoding on the unit circle and harmonic coherence as a universal relationship detection operator. A single function — `cos(n * (θ_a - θ_b))` — detects exact matches, harmonic families, oppositions, fuzzy proximity, and multi-type relationships, matching or exceeding the expressiveness of traditional WHERE and JOIN operations for relationship-dense queries.

The framework includes:

- **A geometric relationship catalog** — every structural relationship pattern discoverable on a phase circle, stripped of all domain-specific interpretation, expressed as pure mathematics
- **A validation paper** — 13 tests, 3 corrective findings, all passing, with reproducible Rust code
- **An architecture proposal** — applying wave mechanics as a substrate for LLM attention and knowledge representation

## Origin

This work emerged from an unconventional observation: multiple ancient traditions independently discovered the same geometric relationship patterns by dividing circles into segments and cataloging which angles produce meaningful relationships. When stripped of interpretive layers, what remains is a complete taxonomy of relationship types — symmetric, asymmetric, directed, structural, context-dependent, compound — unified on a single mathematical substrate.

The key theoretical insights:

1. **Harmonic coherence (an established Fourier operation) works as a universal relationship operator for database queries.** One function, parameterized by frequency, detects any angular relationship. No relationship-specific code paths needed.
2. **Some relationships are non-geometric.** Structural pairings exist independent of angular distance, requiring explicit lookup tables alongside the geometric engine.
3. **Harmonics are infinite.** The relationship detection capacity is unbounded — not limited to a fixed set of patterns. The geometric invariants (symmetry groups on circles) remain valid at every frequency.
4. **The same primitives that work for database queries map onto LLM attention mechanisms.** Transformers effectively discover wave-like structure through training. Pre-building that structure as the computational substrate could improve efficiency and reasoning capability.

## What Is and Is Not New

**Not new (established mathematics):**
- The equation `cos(n × (θ_a - θ_b))` — this is harmonic coherence, a standard operation in Fourier analysis, known since the 1800s
- Phase encoding values on the unit circle — standard technique in signal processing
- Fourier uniqueness — the theorem that distinct functions have distinct Fourier coefficients
- Cosine similarity as a comparison measure — widely used across many fields

**Potentially new (the application and synthesis):**
- Using harmonic coherence as a database query operator, replacing JOINs with frequency-parameterized scans
- The geometric relationship catalog — a comprehensive taxonomy of relationship types (symmetric, asymmetric, directed, structural, compound) derived from cross-civilizational analysis of circle-division systems, stripped of interpretive layers
- Harmonic fingerprinting for collision resolution — using multi-harmonic probing to disambiguate phase-encoded values, with a validated closed-form formula: `n = ⌈arccos(t) / Δθ⌉`
- The proposal that these primitives could serve as a substrate for LLM attention mechanisms

We make no claim of having discovered new mathematics. The contribution, if any, is in recognising that established mathematical tools solve a specific class of problems (relationship-dense queries) more elegantly than the methods currently used, and in compiling the relationship type catalog that defines what the tools can express.

## Documents

| File | Description |
|------|-------------|
| `docs/geometric-relationship-catalog.md` | Complete catalog of geometric relationship patterns across all source traditions (5 traditions, 26 division systems, 35+ relationship types) |
| `docs/wave-mechanics-stripped-catalog.md` | Pure mathematical specification — all domain-specific interpretation removed, only structural geometry remains |
| `docs/wave-test-program.md` | Test program specification — 13 tests validating the core math |
| `docs/wave-mechanics-validation-paper-theoretical.md` | Pre-test validation paper — formal framework and expected results (written before code execution) |
| `docs/wave-mechanics-validation-paper-empirical.md` | Post-test validation paper — actual results, real numbers, three corrective findings from running the code |
| `src/` | Rust source code for the validation test suite (~800 lines, zero dependencies) |

## Reproduce the Validation

```bash
git clone <this-repo>
cd wave-coherence-computational-primitive
cargo run
```

Expected output:

```
=== Wave Mechanics Test Program ===

Test 1:  PASS  (Exact match, zero false positives)
Test 2:  PASS  (3rd harmonic detects 0°, 120°, 240°)
Test 3:  PASS  (2nd harmonic detects 0°, 180°)
Test 4:  PASS  (Fuzzy scores: 1.000 > 0.924 > 0.556 > 0.0)
Test 5:  PASS  (Multi-attribute AND via product)
Test 6:  PASS  (All 4 directed cycle traversals correct)
Test 7:  PASS  (Structural pairs independent of geometry)
Test 8:  PASS  (Wave scan = linear scan, 10/10 matches identical)
Test 9:  PASS  (Single scan found 75 entities across 3 groups)
Test 10: PASS  (Broad: 3 targets, Narrow: 1 target, same position)
Test 11: PASS  (Harmonic fingerprinting: predicted n matches actual at 2°, 1°, 0.1°)
Test 12: PASS  (Mutual amplification: ordering and ratios exact)
Test 13: PASS  (5-node cycle: 20/20 pairs, 4 types × 5, zero conflicts)

=== RESULTS: 13 passed, 0 failed out of 13 ===
ALL TESTS PASSED
```

Requires only a Rust toolchain (edition 2024). No external dependencies.

## Key Results

**Test 9 is the critical result.** A single harmonic coherence scan (`cos(3 * delta)`) found 75 related entities across 3 groups in one pass. The equivalent SQL requires 2-3 JOINs and an explicit relation table. The wave model discovers relationships from geometry; the relational model must enumerate them.

**Test 11 validates harmonic fingerprinting.** Bucket collisions are resolvable by probing higher harmonics. The required harmonic has a closed-form formula: `n = ⌈arccos(t) / Δθ⌉`. Predicted matched actual exactly at 2° (n=13), 1° (n=26), and 0.1° (n=259). Collision resolution scales by analysis depth, not storage.

**Three corrective findings tighten the design:**

1. **Bucket resolution imposes a threshold floor.** Exact match threshold must exceed `cos(2π / bucket_count)` to avoid neighbor leakage. Analogous to the Nyquist limit in signal processing.
2. **Cosine orb falloff is nonlinear.** At 62.5% of tolerance radius, score is 0.556 (not ~0.7). The curve is concave — generous near center, steep near edge.
3. **Asymmetric operations require directed distance.** Shortest-path distance (0-180°) destroys directionality. Typed reach needs directed distance (0-360°).

## Potential Applications

### Database Query Engine
Phase-encoded entities with coherence-based scanning for relationship-dense data. Compound relational queries (harmonic family + structural pair + directed dependency + domain relevance) computed as interference patterns rather than multiple JOINs.

### LLM Architecture
- Attention heads parameterized by harmonic frequency instead of learned weights
- Multi-resolution positional encoding matching the tier system
- Context windows as resonance fields where relevance emerges from constructive interference
- Directed phase relationships as native reasoning chain primitives
- Infinite harmonic capacity without additional parameters

### Knowledge Graph / RAG
Typed retrieval that surfaces not just "documents about X" but "documents about things that enable X" or "documents about things X conflicts with" — relationship-typed retrieval that cosine similarity alone cannot express.

## Related Work

Listopad (2025) independently developed ResonanceDB, a phase-aware retrieval system that scores document relevance using resonance-based coherence rather than cosine similarity over flat embeddings. Their empirical results validate that phase-encoded scoring outperforms standard vector retrieval for relationship-sensitive queries. The present work extends this direction from the retrieval layer to the encoding substrate itself — proposing harmonic coherence not as a scoring alternative bolted onto existing embeddings, but as the foundational computational primitive for encoding, querying, and discovering relationships.

- Listopad, S. (2025). *Wave-Based Semantic Memory: A Phase-Aware Alternative to Vector Retrieval.* arXiv:2509.09691. https://arxiv.org/abs/2509.09691

## Attribution

This work is a collaboration between Marco (conceptual framework, key theoretical insights, architectural direction) and Claude (Anthropic's AI assistant — mathematical formalization, documentation, test design, and code generation). 

The core insights — that ancient geometric relationship catalogs encode a complete taxonomy of structural relationships, that harmonics are infinite and geometric invariants persist across all frequencies, and that these primitives map onto LLM attention mechanisms — originated from Marco's observations and questions during extended collaborative sessions.

## License

All documents, code, and specifications in this repository are released under dual license:

- **Code:** MIT License
- **Documents:** Creative Commons Attribution 4.0 International (CC BY 4.0)

This work is published as prior art to ensure it remains freely available and unpatentable. Use it, extend it, build on it, commercialize implementations of it. The ideas belong to everyone.

## Why Open

This work is released openly and freely because the originators believe that if these mathematical primitives are genuinely useful — for databases, for LLM architectures, for knowledge representation — they should be available to everyone, not locked behind patents or proprietary implementations. 

Publishing establishes prior art. Prior art prevents patents. What belongs to everyone cannot be taken by anyone.

---

*"The patterns are mathematical facts, not cultural inventions. Every civilization that studied circles found the same ones."*
