"""
Harmonic Structure Analysis of Real Model Embeddings

Tests whether real transformer embeddings contain harmonic structure
that standard cosine similarity cannot detect -- the hypothesis from
Test 21, applied to real model vectors instead of synthetic ones.

Uses sentence-transformers all-MiniLM-L6-v2 (384 dimensions, ~80MB).
Zero dependencies beyond sentence-transformers + numpy (both pulled
by sentence-transformers).

Analysis approach:
  1. Standard cosine similarity (baseline)
  2. Per-dimension product analysis (cancellation detection)
  3. FFT spectral analysis (frequency structure in embedding vectors)
  4. Block cosine similarity (partial-dimension relationship typing)
  5. Harmonic coherence on phase-encoded dimensions
"""

import math
import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# Helper functions
# =============================================================================

def cosine_similarity(a, b):
    """Standard cosine similarity."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


def element_wise_products(a, b):
    """Per-dimension product (what cosine similarity sums over)."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return a_norm * b_norm


def block_cosine(a, b, n_blocks):
    """Cosine similarity computed independently per block of dimensions."""
    dim = len(a)
    block_size = dim // n_blocks
    results = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else dim
        block_a = a[start:end]
        block_b = b[start:end]
        norm_a = np.linalg.norm(block_a)
        norm_b = np.linalg.norm(block_b)
        if norm_a > 0 and norm_b > 0:
            results.append(np.dot(block_a, block_b) / (norm_a * norm_b))
        else:
            results.append(0.0)
    return results


def fft_magnitude(vec):
    """FFT magnitude spectrum of an embedding vector."""
    spectrum = np.fft.rfft(vec)
    return np.abs(spectrum)


def fft_phase(vec):
    """FFT phase spectrum of an embedding vector."""
    spectrum = np.fft.rfft(vec)
    return np.angle(spectrum)


def spectral_coherence(a, b, n_bands=12):
    """
    Coherence between two vectors computed per frequency band.
    Treats each embedding as a signal, decomposes via FFT, then
    measures alignment per band rather than globally.
    """
    spec_a = np.fft.rfft(a)
    spec_b = np.fft.rfft(b)
    n_freq = len(spec_a)
    band_size = n_freq // n_bands

    results = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n_freq
        band_a = spec_a[start:end]
        band_b = spec_b[start:end]

        # Cross-spectral coherence per band
        cross = np.sum(band_a * np.conj(band_b))
        power_a = np.sum(np.abs(band_a) ** 2)
        power_b = np.sum(np.abs(band_b) ** 2)

        if power_a > 0 and power_b > 0:
            coh = np.real(cross) / math.sqrt(power_a * power_b)
        else:
            coh = 0.0
        results.append(coh)

    return results


def cancellation_ratio(products):
    """
    Measures how much cancellation occurs in the cosine similarity sum.
    Ratio = |sum(products)| / sum(|products|)
    = 1.0 means no cancellation (all same sign)
    = 0.0 means complete cancellation (positive = negative)
    """
    abs_sum = np.sum(np.abs(products))
    signed_sum = abs(np.sum(products))
    if abs_sum == 0:
        return 0.0
    return signed_sum / abs_sum


# =============================================================================
# Word groups with known semantic relationships
# =============================================================================

WORD_GROUPS = {
    "synonyms": [
        ("happy", "joyful"),
        ("sad", "sorrowful"),
        ("fast", "quick"),
        ("big", "large"),
    ],
    "antonyms": [
        ("happy", "sad"),
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
    ],
    "hierarchical": [
        ("animal", "dog"),
        ("fruit", "apple"),
        ("vehicle", "car"),
        ("color", "red"),
    ],
    "functional": [
        ("doctor", "hospital"),
        ("teacher", "school"),
        ("pilot", "airplane"),
        ("chef", "kitchen"),
    ],
    "analogical": [
        # king:queen :: man:woman
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
    ],
    "unrelated": [
        ("banana", "democracy"),
        ("guitar", "volcano"),
        ("pencil", "hurricane"),
        ("sofa", "algebra"),
    ],
}


# =============================================================================
# Main analysis
# =============================================================================

def main():
    print("=" * 70)
    print("  Harmonic Structure Analysis of Real Model Embeddings")
    print("  Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")
    print()

    # Collect all unique words
    all_words = set()
    for pairs in WORD_GROUPS.values():
        for w1, w2 in pairs:
            all_words.add(w1)
            all_words.add(w2)
    all_words = sorted(all_words)

    # Generate embeddings
    print(f"Generating embeddings for {len(all_words)} words...")
    embeddings = model.encode(all_words)
    word_to_emb = {w: embeddings[i] for i, w in enumerate(all_words)}
    print(f"Embedding shape: {embeddings[0].shape}")
    print()

    # =========================================================================
    # Analysis 1: Cosine Similarity Baseline
    # =========================================================================
    print("-" * 70)
    print("  Analysis 1: Standard Cosine Similarity (Baseline)")
    print("-" * 70)
    print()

    for group_name, pairs in WORD_GROUPS.items():
        print(f"  {group_name}:")
        for w1, w2 in pairs:
            cs = cosine_similarity(word_to_emb[w1], word_to_emb[w2])
            print(f"    {w1:>12} -- {w2:<12}  cosine = {cs:.4f}")
        group_mean = np.mean([
            cosine_similarity(word_to_emb[w1], word_to_emb[w2])
            for w1, w2 in pairs
        ])
        print(f"    {'':>12}    {'group mean':>12}  cosine = {group_mean:.4f}")
        print()

    # =========================================================================
    # Analysis 2: Cancellation Detection
    # =========================================================================
    print("-" * 70)
    print("  Analysis 2: Cancellation Ratio")
    print("  (1.0 = no cancellation, 0.0 = complete cancellation)")
    print("  If ratio << 1, cosine similarity is losing per-dimension info")
    print("-" * 70)
    print()

    for group_name, pairs in WORD_GROUPS.items():
        print(f"  {group_name}:")
        for w1, w2 in pairs:
            prods = element_wise_products(word_to_emb[w1], word_to_emb[w2])
            cr = cancellation_ratio(prods)
            cs = cosine_similarity(word_to_emb[w1], word_to_emb[w2])
            pos_sum = np.sum(prods[prods > 0])
            neg_sum = np.sum(prods[prods < 0])
            print(f"    {w1:>12} -- {w2:<12}  cancel_ratio = {cr:.4f}  "
                  f"(+{pos_sum:.3f} / {neg_sum:.3f} = cosine {cs:.4f})")
        print()

    # =========================================================================
    # Analysis 3: Block Cosine Similarity
    # =========================================================================
    n_blocks = 8
    print("-" * 70)
    print(f"  Analysis 3: Block Cosine Similarity ({n_blocks} blocks of ~{384 // n_blocks} dims)")
    print("  Tests if different dimension blocks capture different relationship types")
    print("-" * 70)
    print()

    for group_name, pairs in WORD_GROUPS.items():
        print(f"  {group_name}:")
        for w1, w2 in pairs:
            blocks = block_cosine(word_to_emb[w1], word_to_emb[w2], n_blocks)
            block_str = " ".join(f"{b:>6.3f}" for b in blocks)
            cs = cosine_similarity(word_to_emb[w1], word_to_emb[w2])
            variance = np.var(blocks)
            print(f"    {w1:>12} -- {w2:<12}  blocks: [{block_str}]  var={variance:.4f}  cos={cs:.4f}")
        print()

    # =========================================================================
    # Analysis 4: Spectral Coherence Per Band
    # =========================================================================
    n_bands = 8
    print("-" * 70)
    print(f"  Analysis 4: Spectral Coherence ({n_bands} frequency bands)")
    print("  FFT decomposes each embedding into frequency components,")
    print("  then measures coherence per band -- like our harmonic sweep")
    print("-" * 70)
    print()

    for group_name, pairs in WORD_GROUPS.items():
        print(f"  {group_name}:")
        for w1, w2 in pairs:
            bands = spectral_coherence(word_to_emb[w1], word_to_emb[w2], n_bands)
            band_str = " ".join(f"{b:>6.3f}" for b in bands)
            cs = cosine_similarity(word_to_emb[w1], word_to_emb[w2])
            variance = np.var(bands)
            print(f"    {w1:>12} -- {w2:<12}  bands: [{band_str}]  var={variance:.4f}  cos={cs:.4f}")
        print()

    # =========================================================================
    # Analysis 5: Relationship Discrimination
    # =========================================================================
    print("-" * 70)
    print("  Analysis 5: Relationship Type Discrimination")
    print("  Can any method distinguish synonyms from antonyms from unrelated?")
    print("  (Cosine similarity notoriously cannot for antonyms)")
    print("-" * 70)
    print()

    # Compute mean metrics per group
    metrics = {}
    for group_name, pairs in WORD_GROUPS.items():
        cosines = []
        cancel_ratios = []
        block_variances = []
        spectral_variances = []
        spectral_profiles = []

        for w1, w2 in pairs:
            a, b = word_to_emb[w1], word_to_emb[w2]
            cosines.append(cosine_similarity(a, b))
            prods = element_wise_products(a, b)
            cancel_ratios.append(cancellation_ratio(prods))
            blocks = block_cosine(a, b, n_blocks)
            block_variances.append(np.var(blocks))
            bands = spectral_coherence(a, b, n_bands)
            spectral_variances.append(np.var(bands))
            spectral_profiles.append(bands)

        metrics[group_name] = {
            "cosine_mean": np.mean(cosines),
            "cosine_std": np.std(cosines),
            "cancel_mean": np.mean(cancel_ratios),
            "block_var_mean": np.mean(block_variances),
            "spectral_var_mean": np.mean(spectral_variances),
            "spectral_profile": np.mean(spectral_profiles, axis=0),
        }

    print(f"  {'Group':<15} {'Cosine':>10} {'Cancel':>10} {'Block Var':>10} {'Spec Var':>10}")
    print(f"  {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for group_name, m in metrics.items():
        print(f"  {group_name:<15} {m['cosine_mean']:>10.4f} {m['cancel_mean']:>10.4f} "
              f"{m['block_var_mean']:>10.4f} {m['spectral_var_mean']:>10.4f}")
    print()

    # =========================================================================
    # Analysis 6: Spectral Fingerprint Comparison
    # =========================================================================
    print("-" * 70)
    print("  Analysis 6: Mean Spectral Profile Per Relationship Type")
    print("  If different relationship types have different spectral shapes,")
    print("  that is structure cosine similarity destroys by summing.")
    print("-" * 70)
    print()

    print(f"  {'Group':<15}", end="")
    for i in range(n_bands):
        print(f"  Band {i}", end="")
    print()
    print(f"  {'-' * 15}", end="")
    for _ in range(n_bands):
        print(f"  {'------':>6}", end="")
    print()

    for group_name, m in metrics.items():
        profile = m["spectral_profile"]
        print(f"  {group_name:<15}", end="")
        for v in profile:
            print(f"  {v:>6.3f}", end="")
        print()
    print()

    # =========================================================================
    # Analysis 7: Analogy Vector Test
    # =========================================================================
    print("-" * 70)
    print("  Analysis 7: Analogy Vectors (king-man+woman ~ queen)")
    print("  Tests if directional relationships exist in embedding space")
    print("-" * 70)
    print()

    # Classic analogy: king - man + woman =? queen
    king = word_to_emb["king"]
    man = word_to_emb["man"]
    woman = word_to_emb["woman"]
    queen = word_to_emb["queen"]

    analogy_vec = king - man + woman

    # Find cosine similarity of analogy vector to all words
    print("  king - man + woman -> closest words:")
    similarities = []
    for w in all_words:
        cs = cosine_similarity(analogy_vec, word_to_emb[w])
        similarities.append((w, cs))
    similarities.sort(key=lambda x: -x[1])
    for w, cs in similarities[:10]:
        marker = " <-- target" if w == "queen" else ""
        print(f"    {w:<15} cosine = {cs:.4f}{marker}")
    print()

    # Now check: does the gender direction (woman - man) have spectral structure?
    gender_dir = woman - man
    gender_fft = fft_magnitude(gender_dir)
    print("  Gender direction (woman - man) FFT magnitude (first 20 components):")
    print("   ", " ".join(f"{v:.3f}" for v in gender_fft[:20]))
    print()

    # Compare FFT of king-queen direction vs boy-girl direction
    kq_dir = queen - king
    bg_dir = word_to_emb["girl"] - word_to_emb["boy"]
    fm_dir = word_to_emb["mother"] - word_to_emb["father"]

    # Spectral coherence between these relationship vectors
    print("  Spectral coherence between relationship direction vectors:")
    pairs_to_test = [
        ("queen-king", "girl-boy", kq_dir, bg_dir),
        ("queen-king", "mother-father", kq_dir, fm_dir),
        ("girl-boy", "mother-father", bg_dir, fm_dir),
        ("queen-king", "woman-man", kq_dir, gender_dir),
    ]
    for name1, name2, v1, v2 in pairs_to_test:
        cs = cosine_similarity(v1, v2)
        bands = spectral_coherence(v1, v2, n_bands)
        band_str = " ".join(f"{b:.3f}" for b in bands)
        print(f"    {name1:>15} vs {name2:<15}  cosine={cs:.4f}  bands=[{band_str}]")
    print()

    # =========================================================================
    # Analysis 8: Dimension Cluster Entropy
    # =========================================================================
    print("-" * 70)
    print("  Analysis 8: Product Sign Pattern Analysis")
    print("  Do antonyms and synonyms have systematically different")
    print("  patterns of positive/negative dimension products?")
    print("-" * 70)
    print()

    for group_name in ["synonyms", "antonyms", "unrelated"]:
        pairs = WORD_GROUPS[group_name]
        pos_fracs = []
        for w1, w2 in pairs:
            prods = element_wise_products(word_to_emb[w1], word_to_emb[w2])
            pos_frac = np.mean(prods > 0)
            pos_fracs.append(pos_frac)
        mean_pos = np.mean(pos_fracs)
        print(f"  {group_name:<12}: mean fraction of positive-product dimensions = {mean_pos:.4f}")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    # Key finding: is there structure beyond cosine similarity?
    syn_cos = metrics["synonyms"]["cosine_mean"]
    ant_cos = metrics["antonyms"]["cosine_mean"]
    unr_cos = metrics["unrelated"]["cosine_mean"]

    print(f"  Cosine similarity averages:")
    print(f"    Synonyms:    {syn_cos:.4f}")
    print(f"    Antonyms:    {ant_cos:.4f}")
    print(f"    Unrelated:   {unr_cos:.4f}")
    print()

    if ant_cos > 0.3:
        print("  KEY FINDING: Cosine similarity rates antonyms as SIMILAR (> 0.3).")
        print("  This is the known cosine similarity blind spot -- it cannot")
        print("  distinguish 'related-same' from 'related-opposite'.")
    else:
        print("  Antonyms show low cosine similarity in this model.")
    print()

    syn_cancel = metrics["synonyms"]["cancel_mean"]
    ant_cancel = metrics["antonyms"]["cancel_mean"]
    unr_cancel = metrics["unrelated"]["cancel_mean"]

    print(f"  Cancellation ratios:")
    print(f"    Synonyms:    {syn_cancel:.4f}")
    print(f"    Antonyms:    {ant_cancel:.4f}")
    print(f"    Unrelated:   {unr_cancel:.4f}")
    print()

    if ant_cancel < syn_cancel:
        print("  Antonyms show MORE cancellation than synonyms.")
        print("  This means antonym embeddings have more opposed dimension blocks --")
        print("  structure that cosine similarity sums away.")
    print()

    syn_spec = metrics["synonyms"]["spectral_var_mean"]
    ant_spec = metrics["antonyms"]["spectral_var_mean"]
    unr_spec = metrics["unrelated"]["spectral_var_mean"]

    print(f"  Spectral variance:")
    print(f"    Synonyms:    {syn_spec:.4f}")
    print(f"    Antonyms:    {ant_spec:.4f}")
    print(f"    Unrelated:   {unr_spec:.4f}")
    print()

    print("  If spectral variance differs by relationship type, that is")
    print("  per-frequency structure that cosine similarity (a single global")
    print("  dot product) cannot capture -- precisely the phenomenon")
    print("  demonstrated in Test 21 with synthetic vectors.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
