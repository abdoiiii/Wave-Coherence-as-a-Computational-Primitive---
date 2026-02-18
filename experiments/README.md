# Experiments

Exploratory experiments on a 4-layer, 4-head, 128-dim harmonic transformer trained on Shakespeare (~1.1M chars, 842K parameters, 65 tokens). All experiments run with CUDA, 3000 training steps, batch 64, lr 3e-4. Results are from a small model — findings should scale with model size.

## Scripts

| # | File | Phase |
|---|------|-------|
| 1 | `spectral_persistence.py` | Spectral persistence through layers |
| 2 | `geometric_relations_probe.py` | Geometric relations between harmonic channels |
| 3 | `knowledge_editing.py` | Knowledge editing via weight surgery |
| 3b | `harmonic_injection.py`, `harmonic_injection_v2.py` | Knowledge editing via embedding injection |
| 4 | `harmonic_construction.py` | Constructing novel harmonic vectors |
| 5 | `musical_harmonics.py` | Musical interval predictions |
| 6 | `progressive_learning.py` | Structure-first training curriculum |
| 7 | `concept_composition.py` | Character-to-word composition |
| 8 | `init_convergence.py` | Cross-seed structural convergence |
| 9 | `commitment_point.py` | Layer-wise decision commitment |
| 10 | `early_exit.py` | Token-level early exit |
| 11 | `chord_flow.py` | Dynamic token merging |
| 12 | `natural_expression.py` | Internal representation landscape |
| 13 | `expression_curriculum.py` | Output head comparison |
| 14 | `shakespeare_knowledge.py` | Knowledge probing |
| 15 | `harmonic_decoder.py` | Harmonic-aware decoding |
| — | `sweep-test/` | Rust harmonic sweep validation (Test 21) |

## Results

See [RESULTS.md](RESULTS.md) for data tables.
