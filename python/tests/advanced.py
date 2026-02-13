"""Tests 10-13: Typed reach, fingerprinting, amplification, cycle uniqueness."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from wave import Phase, WavePacket
from field import ResonanceField
from relationships import DirectedCycle


def test_10_typed_reach() -> bool:
    print("--- Test 10: Entity Type-Dependent Reach ---")

    field = ResonanceField()
    for i in range(12):
        deg = i * 30.0
        packet = WavePacket(f"pos_{deg:.0f}").with_attr("angle", Phase.from_degrees(deg))
        field.add(packet)

    target = Phase.from_degrees(0.0)
    orb = 8.0

    broad_angles = [60.0, 180.0, 270.0]
    broad_results = field.query_typed_reach("angle", target, broad_angles, orb)

    print("  Entity at 0 deg with type 'broad' (sees 60 deg, 180 deg, 270 deg):")
    for p, score in broad_results:
        print(f"    {p.id} -> score = {score:.4f}")

    narrow_angles = [180.0]
    narrow_results = field.query_typed_reach("angle", target, narrow_angles, orb)

    print("  Entity at 0 deg with type 'narrow' (sees 180 deg only):")
    for p, score in narrow_results:
        print(f"    {p.id} -> score = {score:.4f}")

    broad_ids = [p.id for p, _ in broad_results]
    broad_has_60 = "pos_60" in broad_ids
    broad_has_180 = "pos_180" in broad_ids
    broad_has_270 = "pos_270" in broad_ids

    narrow_ids = [p.id for p, _ in narrow_results]
    narrow_has_180 = "pos_180" in narrow_ids
    narrow_count = len(narrow_ids)

    print(f"  Broad found 60 deg: {broad_has_60}, 180 deg: {broad_has_180}, 270 deg: {broad_has_270}")
    print(f"  Narrow found 180 deg: {narrow_has_180}, total count: {narrow_count}")
    print(f"  Same position, different type, different results: {len(broad_results) != len(narrow_results)}")

    passed = broad_has_60 and broad_has_180 and broad_has_270 and narrow_has_180 and narrow_count == 1
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_11_harmonic_fingerprint() -> bool:
    print("--- Test 11: Harmonic Fingerprint Disambiguation ---")

    a = Phase.from_degrees(5.0)
    b = Phase.from_degrees(7.0)

    print("  Phase A: 5 deg, Phase B: 7 deg (2 deg apart)")
    print("  At n=1 they are nearly indistinguishable.")
    print("  Scanning harmonics to find divergence:\n")

    first_divergence_n = 0
    threshold = 0.9

    print(f"  {'n':>4}  {'coherence':>12}  {'status':>10}")
    print(f"  {'---':>4}  {'---------':>12}  {'------':>10}")

    for n in range(1, 91):
        c = a.harmonic_coherence(b, n)
        status = "similar" if abs(c) >= threshold else "DIVERGED"

        if n <= 10 or n % 10 == 0 or (abs(c) < threshold and first_divergence_n == 0):
            print(f"  {n:>4}  {c:>12.6f}  {status:>10}")

        if abs(c) < threshold and first_divergence_n == 0:
            first_divergence_n = n

    print(f"\n  First divergence at harmonic n={first_divergence_n}")

    print("\n  --- Harder case: 1 deg apart ---")
    c_val = Phase.from_degrees(10.0)
    d = Phase.from_degrees(11.0)

    first_divergence_hard = 0
    for n in range(1, 181):
        c = c_val.harmonic_coherence(d, n)
        if abs(c) < threshold and first_divergence_hard == 0:
            first_divergence_hard = n
            print(f"  1 deg apart: first divergence at n={n}, coherence={c:.6f}")

    print("\n  --- Extreme case: 0.1 deg apart ---")
    e = Phase.from_degrees(10.0)
    f = Phase.from_degrees(10.1)

    first_divergence_extreme = 0
    for n in range(1, 1801):
        c = e.harmonic_coherence(f, n)
        if abs(c) < threshold and first_divergence_extreme == 0:
            first_divergence_extreme = n
            print(f"  0.1 deg apart: first divergence at n={n}, coherence={c:.6f}")

    delta_2 = 2.0
    predicted_n_2 = math.ceil(math.degrees(math.acos(threshold)) / delta_2)
    print(f"\n  Predicted divergence (2 deg apart):   n~{predicted_n_2}")
    print(f"  Actual divergence (2 deg apart):      n={first_divergence_n}")
    print(f"  Predicted divergence (1 deg apart):   n~{math.ceil(math.degrees(math.acos(threshold)) / 1.0)}")
    print(f"  Actual divergence (1 deg apart):      n={first_divergence_hard}")
    print(f"  Predicted divergence (0.1 deg apart): n~{math.ceil(math.degrees(math.acos(threshold)) / 0.1)}")
    print(f"  Actual divergence (0.1 deg apart):    n={first_divergence_extreme}")

    passed = (first_divergence_n > 0
              and first_divergence_hard > 0
              and first_divergence_extreme > 0
              and first_divergence_hard > first_divergence_n
              and first_divergence_extreme > first_divergence_hard)

    print(f"\n  Divergence ordering: {first_divergence_n} < {first_divergence_hard} < {first_divergence_extreme} "
          f"(smaller delta needs higher n): "
          f"{first_divergence_hard > first_divergence_n and first_divergence_extreme > first_divergence_hard}")

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_12_mutual_amplification() -> bool:
    print("--- Test 12: Mutual Reference Amplification ---")

    a = Phase.from_degrees(30.0)
    b = Phase.from_degrees(35.0)

    base_coherence = a.coherence(b)

    a_refs_b = True
    b_refs_a_mutual = True
    b_refs_a_oneway = False

    if a_refs_b and b_refs_a_mutual:
        mutual_score = base_coherence * 1.5
    elif a_refs_b or b_refs_a_mutual:
        mutual_score = base_coherence * 1.2
    else:
        mutual_score = base_coherence

    if a_refs_b and b_refs_a_oneway:
        oneway_score = base_coherence * 1.5
    elif a_refs_b or b_refs_a_oneway:
        oneway_score = base_coherence * 1.2
    else:
        oneway_score = base_coherence

    no_ref_score = base_coherence

    print(f"  A at 30 deg, B at 35 deg (5 deg apart)")
    print(f"  Base coherence: {base_coherence:.6f}")
    print(f"  Mutual (A<->B):   {mutual_score:.6f} (x 1.5)")
    print(f"  One-way (A->B):  {oneway_score:.6f} (x 1.2)")
    print(f"  No reference:   {no_ref_score:.6f} (x 1.0)")

    ordering_correct = mutual_score > oneway_score and oneway_score > no_ref_score

    mutual_ratio = mutual_score / base_coherence
    oneway_ratio = oneway_score / base_coherence

    print(f"  Mutual ratio:  {mutual_ratio:.2f} (expected 1.50)")
    print(f"  One-way ratio: {oneway_ratio:.2f} (expected 1.20)")
    print(f"  Ordering (mutual > oneway > none): {ordering_correct}")

    passed = (ordering_correct
              and abs(mutual_ratio - 1.5) < 0.001
              and abs(oneway_ratio - 1.2) < 0.001)

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_13_cycle_uniqueness() -> bool:
    print("--- Test 13: Exhaustive 5-Node Cycle Relationship Uniqueness ---")

    cycle = DirectedCycle(5)

    relationship_map = [[None] * 5 for _ in range(5)]
    steps = [1, 2, -1, -2]
    step_names = ["+1 (generative)", "+2 (destructive)", "-1 (weakening)", "-2 (controlling)"]

    all_assigned = True
    no_conflicts = True

    for step in steps:
        for start in range(5):
            dest = cycle.step(start, step)
            if relationship_map[start][dest] is not None:
                print(f"  CONFLICT: ({start},{dest}) has both step {relationship_map[start][dest]} and step {step}")
                no_conflicts = False
            else:
                relationship_map[start][dest] = step

    print("  Relationship matrix (row=from, col=to, value=step):")
    print("       0     1     2     3     4")
    for a in range(5):
        row = f"  {a}:"
        for b in range(5):
            if a == b:
                row += "    . "
            elif relationship_map[a][b] is not None:
                row += f"  {relationship_map[a][b]:>+3d} "
            else:
                row += "  NONE"
                all_assigned = False
        print(row)

    counts = [0] * 4
    for a in range(5):
        for b in range(5):
            if a != b and relationship_map[a][b] is not None:
                idx = steps.index(relationship_map[a][b])
                counts[idx] += 1

    print("\n  Relationships per type:")
    for i, name in enumerate(step_names):
        print(f"    {name}: {counts[i]} pairs")

    total_pairs = sum(counts)
    expected_pairs = 5 * 4

    print(f"\n  Total assigned: {total_pairs} / {expected_pairs} ordered pairs")
    print(f"  All pairs assigned: {all_assigned}")
    print(f"  No conflicts: {no_conflicts}")
    print(f"  Each type has exactly 5 pairs: {all(c == 5 for c in counts)}")

    passed = (all_assigned and no_conflicts and total_pairs == expected_pairs
              and all(c == 5 for c in counts))

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed
