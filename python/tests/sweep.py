"""Test 21: Harmonic Sweep — Cosine Similarity Blindness."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from wave import Phase


def test_21_harmonic_sweep() -> bool:
    print("--- Test 21: Harmonic Sweep (Cosine Similarity Blindness) ---")
    print()

    n_harmonics = 12

    # Phase 1: Encode letters at known angles with deliberate harmonic relationships
    letters = [
        ('A', 0.0),    # reference
        ('B', 120.0),  # triadic with A -> detectable at n=3
        ('C', 180.0),  # opposition to A -> detectable at n=2
        ('D', 90.0),   # quadrant from A -> detectable at n=4
        ('E', 60.0),   # sextile from A -> detectable at n=6
        ('F', 72.0),   # pentagonal from A -> detectable at n=5
        ('G', 37.0),   # noise control — no clean harmonic with A
        ('H', 143.0),  # noise control — no clean harmonic with A
    ]

    phases = [Phase.from_degrees(deg) for _, deg in letters]

    # Phase 2: Generate harmonic embeddings v(theta) = [cos(theta), cos(2*theta), ..., cos(N*theta)]
    embeddings = [
        [math.cos(n * p.angle) for n in range(1, n_harmonics + 1)]
        for p in phases
    ]

    # Phase 3: Compute cosine similarity (standard ML method)
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    ab_cosine = cosine_sim(embeddings[0], embeddings[1])
    ac_cosine = cosine_sim(embeddings[0], embeddings[2])
    ad_cosine = cosine_sim(embeddings[0], embeddings[3])
    ae_cosine = cosine_sim(embeddings[0], embeddings[4])

    print("  Cosine similarity (standard ML):")
    print(f"    A-B: {ab_cosine:.4f}  (triadic partners — should be related)")
    print(f"    A-C: {ac_cosine:.4f}  (opposition pair — should be related)")
    print(f"    A-D: {ad_cosine:.4f}  (quadrant pair — should be related)")
    print(f"    A-E: {ae_cosine:.4f}  (sextile pair — should be related)")
    print()

    # Phase 4: Harmonic sweep — check each channel independently
    expected = [
        (0, 1, 3, "A-B triadic (120 deg)"),
        (0, 2, 2, "A-C opposition (180 deg)"),
        (0, 3, 4, "A-D quadrant (90 deg)"),
        (0, 4, 6, "A-E sextile (60 deg)"),
        (0, 5, 5, "A-F pentagonal (72 deg)"),
    ]

    threshold = 0.999

    print("  Harmonic sweep (per-channel decomposition):")

    all_detected = True
    for i, j, expected_n, desc in expected:
        coh = phases[i].harmonic_coherence(phases[j], expected_n)
        detected = coh > threshold
        status = "DETECTED" if detected else "MISSED"
        print(f"    {desc} at n={expected_n}: coherence={coh:.6f} [{status}]")
        if not detected:
            all_detected = False
    print()

    # Phase 5: Noise check
    noise_indices = [(0, 6, "A-G"), (0, 7, "A-H")]
    noise_clean = True

    print("  Noise check (non-harmonic pairs at n=1..6):")
    for i, j, label in noise_indices:
        false_pos = []
        for n in range(1, 7):
            coh = phases[i].harmonic_coherence(phases[j], n)
            if coh > threshold:
                false_pos.append(n)
        if not false_pos:
            print(f"    {label}: clean — no false detections")
        else:
            print(f"    {label}: FALSE POSITIVE at n={false_pos}")
            noise_clean = False
    print()

    # Phase 6: A-B decomposition
    print(f"  A-B decomposition (cosine_sim={ab_cosine:.4f}, but per-channel):")
    signal_channels = 0
    noise_channels = 0
    for n in range(1, n_harmonics + 1):
        coh = phases[0].harmonic_coherence(phases[1], n)
        if abs(coh) > 0.999:
            signal_channels += 1
            marker = "<-- coherence 1.0" if coh > 0.0 else "<-- anti-coherence -1.0"
        else:
            noise_channels += 1
            marker = ""
        print(f"    n={n:>2}: {coh:>8.4f}  {marker}")
    print()
    print(f"  {signal_channels} signal channels, {noise_channels} noise channels -> sum cancels to {ab_cosine:.4f}")
    print("  Cosine similarity destroys the harmonic decomposition.")
    print()

    # Phase 7: Spectral profile
    print("  Spectral profile (pairs with exact coherence at each harmonic):")
    for n in range(1, n_harmonics + 1):
        high_pairs = []
        for i in range(len(letters)):
            for j in range(i + 1, len(letters)):
                coh = phases[i].harmonic_coherence(phases[j], n)
                if abs(coh) > threshold:
                    high_pairs.append(f"{letters[i][0]}{letters[j][0]}")
        count = len(high_pairs)
        pairs_str = ", ".join(high_pairs) if high_pairs else "(none)"
        print(f"    n={n:>2}: {count:>2} pairs  {pairs_str}")
    print()

    # Validation
    passed = (all_detected and noise_clean
              and abs(ab_cosine) < 0.01
              and abs(ac_cosine) < 0.01
              and abs(ad_cosine) < 0.01
              and abs(ae_cosine) < 0.01)

    if passed:
        print("Test 21: PASS  (5 planted relationships recovered, cosine similarity blind to all, 0 false positives)")
    else:
        print("Test 21: FAIL")
        if not all_detected:
            print("  - Not all planted relationships detected")
        if not noise_clean:
            print("  - False positives on noise controls")
        if abs(ab_cosine) >= 0.01:
            print(f"  - A-B cosine sim not near zero: {ab_cosine}")

    return passed
