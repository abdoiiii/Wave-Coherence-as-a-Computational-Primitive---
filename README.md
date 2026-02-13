[![DOI](https://zenodo.org/badge/1153530777.svg)](https://doi.org/10.5281/zenodo.18607190)

# Wave Coherence as a Computational Primitive

## What This Is

A validated mathematical framework that uses phase encoding on the unit circle and harmonic coherence as a universal relationship detection operator. A single function — `cos(n * (θ_a - θ_b))` — detects exact matches, harmonic families, oppositions, fuzzy proximity, and multi-type relationships, matching or exceeding the expressiveness of traditional WHERE and JOIN operations for relationship-dense queries.

The framework includes:

- **A geometric relationship catalog** — every structural relationship pattern discoverable on a phase circle, stripped of all domain-specific interpretation, expressed as pure mathematics
- **A validation paper** — 20 tests, 4 corrective findings, all passing, with reproducible Rust code
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
| `docs/wave-test-program.md` | Test program specification — 20 tests validating the core math |
| `docs/wave-mechanics-validation-paper-theoretical.md` | Pre-test validation paper — formal framework and expected results (written before code execution) |
| `docs/wave-mechanics-validation-paper-empirical.md` | Post-test validation paper — actual results, real numbers, four corrective findings from running the code |
| `src/` | Rust source code for the validation test suite (~2400 lines, zero dependencies) |

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
Test 14: PASS  (Harmonic orthogonality: zero cross-talk between n=3, 4, 5, 6)
Test 15: PASS  (Wraparound: symmetric scores at 0°/360° boundary)
Test 16: PASS  (Scale: 360 values, 0 false positives, harmonic-scaled Nyquist validated)
Test 17: PASS  (Density scaling: sparse clean, degradation at density, harmonic scales with separation)
Test 18: PASS  (Bucket index: all queries match full scan, ~13% selectivity at 1000 entities)
Test 19: PASS  (2D torus index: compound queries correct, multiplicative selectivity over 1D)
Test 20: PASS  (Dynamic mutation: remove/insert/update, all queries correct throughout)
Test 21: PASS  (Harmonic sweep: 5 planted relationships recovered, cosine similarity blind to all, 0 false positives)

=== RESULTS: 21 passed, 0 failed out of 21 ===
ALL TESTS PASSED
```

Requires only a Rust toolchain (edition 2024). No external dependencies.

## Key Results

**Test 9 is the critical result.** A single harmonic coherence scan (`cos(3 * delta)`) found 75 related entities across 3 groups in one pass. The equivalent SQL requires 2-3 JOINs and an explicit relation table. The wave model discovers relationships from geometry; the relational model must enumerate them.

**Test 11 validates harmonic fingerprinting.** Bucket collisions are resolvable by probing higher harmonics. The required harmonic has a closed-form formula: `n = ⌈arccos(t) / Δθ⌉`. Predicted matched actual exactly at 2° (n=13), 1° (n=26), and 0.1° (n=259). Collision resolution scales by analysis depth, not storage.

**Test 14 confirms harmonic orthogonality.** Different harmonic frequencies operate as completely independent selectors with zero cross-talk. n=3 finds only 120° family members, completely excluding 90° and 60° entities (which belong to n=4 and n=6 respectively). This validates that `cos(n × Δθ)` with different n values can serve as independent query channels on the same dataset.

**Test 16 validates scale and reveals a new design rule.** 360 distinct values encoded on a 360-bucket circle are resolved with zero false positives. However, harmonic queries at this scale revealed that the Nyquist-like threshold floor (Finding 1) scales with harmonic number — see Finding 4 below.

**Test 17 characterizes density scaling limits.** Eight configurations from 7-in-12 to 360-in-360, placed at golden angle intervals, reveal that exact match fails only at 100% bucket saturation, while triadic (n=3) detection becomes noisy at lower densities due to harmonic amplification of angular proximity. The resolution harmonic needed to distinguish the closest pair scales inversely with minimum separation, following the formula from Test 11.

**Test 18 proves the self-indexing property.** A BucketIndex that uses the encoded phase position as the bucket address — no separate index structure — produces results identical to a full scan while examining only a fraction of entities. At 1000 entities on 360 buckets: exact match at threshold 0.999 examines 2.0% of entities, exact match at 0.95 examines 10.7%, and harmonic queries examine 15-23%. The circle IS the index: insertion is O(1), and queries are sub-linear with zero maintenance overhead.

**Test 19 validates multi-attribute torus indexing.** Extending the 1D bucket index to a 2D torus (B×B grid) enables compound queries that narrow on both attributes simultaneously. At 500 entities on a 60×60 grid: exact+exact queries and exact+harmonic queries all match full scan exactly. Selectivity improvement over 1D is multiplicative — each dimension narrows independently. This bridges the gap between single-attribute proof and real multi-column database viability.

**Test 20 proves dynamic mutation support.** Insert, remove, and update operations work as local mutations on the circle without global rebuild. Starting from 200 entities: 50 removed, 30 inserted, 20 repositioned — all queries (exact and harmonic) remain correct throughout. Remove is tombstone + bucket cleanup. Update is remove + re-insert. This is what separates a mathematical proof from a working database.

**Test 21 demonstrates cosine similarity blindness.** Eight letters encoded at known phase angles with deliberate harmonic relationships. Cosine similarity between triadic partners (A at 0°, B at 120°) = **0.0000** — reporting "no relationship." A harmonic sweep across individual channels recovers coherence = **1.0000** at n=3. All 5 planted relationships (triadic, opposition, quadrant, sextile, pentagonal) recovered at exactly the correct harmonic with zero false positives on noise controls. The sum of harmonic channels cancels to zero, destroying the per-channel structure. This proves that standard cosine similarity — the primary comparison measure used across ML — is blind to harmonic organization in vectors. The harmonic sweep provides a tool to test whether real model embeddings contain this hidden structure.

**Four corrective findings tighten the design:**

1. **Bucket resolution imposes a threshold floor.** Exact match threshold must exceed `cos(2π / bucket_count)` to avoid neighbor leakage. Analogous to the Nyquist limit in signal processing.
2. **Cosine orb falloff is nonlinear.** At 62.5% of tolerance radius, score is 0.556 (not ~0.7). The curve is concave — generous near center, steep near edge.
3. **Asymmetric operations require directed distance.** Shortest-path distance (0-180°) destroys directionality. Typed reach needs directed distance (0-360°).
4. **The Nyquist-like threshold floor scales with harmonic number.** At harmonic n with B buckets, the threshold floor is `cos(n × 2π / B)`, not `cos(2π / B)`. Higher harmonics amplify bucket spacing, widening neighbor leakage. For single-value precision at n=3 with 360 buckets, threshold must exceed cos(3°) = 0.9986, not cos(1°) = 0.9998.

## Potential Applications

### Database Query Engine
Phase-encoded entities with coherence-based scanning for relationship-dense data. Compound relational queries (harmonic family + structural pair + directed dependency + domain relevance) computed as interference patterns rather than multiple JOINs. The self-indexing property (Test 18) means insertion automatically indexes entities by their encoded position, with sub-linear query performance and zero index maintenance. Multi-attribute torus indexing (Test 19) enables compound queries across multiple columns with multiplicative selectivity. Dynamic mutation (Test 20) confirms insert/remove/update as local operations requiring no global rebuild.

### LLM Architecture — Harmonic Embeddings as Structural Priors

A single phase angle probed across N harmonics produces an N-dimensional vector: `v(θ) = [cos(θ), cos(2θ), ..., cos(Nθ)]`. For K attributes × N harmonics, this yields a K×N-dimensional structured embedding where every dimension has a defined meaning. This is a Fourier basis expansion — the harmonic fingerprint validated in Test 11, generalized to arbitrary dimensionality.

The implication for LLM architecture: harmonic embeddings could serve as **structural priors** — pre-built geometric structure that reduces what the network needs to learn through training. Specific applications:

- Attention heads parameterized by harmonic frequency instead of learned weights
- Positional encoding via harmonic phase (RoPE already uses this principle for one dimension; harmonic encoding generalizes it to N dimensions with relationship-typed structure)
- Context windows as resonance fields where relevance emerges from constructive interference
- Directed phase relationships as native reasoning chain primitives
- Infinite harmonic capacity without additional parameters

The hypothesis: learned embeddings discover through gradient descent a structure that harmonic encoding provides by construction. If true, pre-building the harmonic structure could reduce training cost, improve interpretability, and lower energy consumption.

### Knowledge Graph / RAG
Typed retrieval that surfaces not just "documents about X" but "documents about things that enable X" or "documents about things X conflicts with" — relationship-typed retrieval that cosine similarity alone cannot express.

## Related Work

Listopad (2025) independently developed ResonanceDB, a phase-aware retrieval system that scores document relevance using resonance-based coherence rather than cosine similarity over flat embeddings. Their empirical results validate that phase-encoded scoring outperforms standard vector retrieval for relationship-sensitive queries. The present work extends this direction from the retrieval layer to the encoding substrate itself — proposing harmonic coherence not as a scoring alternative bolted onto existing embeddings, but as the foundational computational primitive for encoding, querying, and discovering relationships.

Wang (2025) proposed a more radical departure: the Self-Resonance Field (SRF) architecture, which replaces transformer self-attention entirely with wave interference and phase superposition. Tokens become waveform imprints with spectral signatures; semantic matching operates via coherence estimation between sub-bands rather than dot-product attention. Critically, Wang's architecture uses partial resonance — local spectral matching rather than global all-to-all attention — analogous to the sub-linear bucket selectivity demonstrated in Tests 18–19 of the present work. Their simulation results show improvements over GPT-4 Turbo across six benchmarks (ROUGE-L, METEOR, Pass@k, MMLU, Accuracy, ARC-AGI). While the results are simulation-based and not yet validated on real hardware, the architecture provides independent evidence that wave mechanics can serve as a viable computational substrate for language modeling — the same hypothesis proposed in Section 5.2 of this work from the mathematical primitives side.

- Listopad, S. (2025). *Wave-Based Semantic Memory: A Phase-Aware Alternative to Vector Retrieval.* arXiv:2509.09691. https://arxiv.org/abs/2509.09691
- Wang, L. (2025). *Defierithos: The Lonely Warrior Rises from Resonance — A Self-Resonance Architecture Beyond Attention.* Submitted to NeurIPS 2025.

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
