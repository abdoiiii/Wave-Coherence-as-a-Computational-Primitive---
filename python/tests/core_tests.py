"""Tests 1-5: Core encoding, harmonics, fuzzy matching, multi-attribute."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wave import Phase, WavePacket
from field import ResonanceField


def test_1_exact_match() -> bool:
    print("--- Test 1: Exact Match (Guessing Game) ---")
    bucket_count = 12
    field = ResonanceField()

    for i in range(12):
        packet = WavePacket(f"entity_{i}").with_attr("value", Phase.from_value(i, bucket_count))
        field.add(packet)

    target = Phase.from_value(7, bucket_count)
    results = field.query_exact("value", target, 0.99)

    print(f"  Target: value=7 on 12-bucket circle")
    print(f"  Matches (coherence > 0.99): {len(results)}")
    for p, c in results:
        print(f"    {p.id} -> coherence = {c:.4f}")

    print("  All coherences:")
    for p in field.packets:
        phase = p.phase_for("value")
        if phase is not None:
            print(f"    {p.id} -> {phase.coherence(target):.4f}")

    passed = len(results) == 1 and results[0][0].id == "entity_7"
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_2_harmonic_family() -> bool:
    print("--- Test 2: Harmonic Family Detection (3rd harmonic / 120 deg) ---")
    field = ResonanceField()

    for i in range(12):
        deg = i * 30.0
        packet = WavePacket(f"pos_{deg:.0f}").with_attr("angle", Phase.from_degrees(deg))
        field.add(packet)

    target = Phase.from_degrees(0.0)
    results = field.query_harmonic("angle", target, 3, 0.95)

    print("  Target: 0 deg, Query: 3rd harmonic, Threshold: 0.95")
    print("  Matches:")
    for p, c in results:
        print(f"    {p.id} -> harmonic_coherence = {c:.4f}")

    print("  All 3rd harmonic coherences:")
    for p in field.packets:
        phase = p.phase_for("angle")
        if phase is not None:
            print(f"    {p.id} -> {phase.harmonic_coherence(target, 3):.4f}")

    expected_ids = ["pos_0", "pos_120", "pos_240"]
    found_ids = [p.id for p, _ in results]
    passed = len(found_ids) == 3 and all(e in found_ids for e in expected_ids)

    print(f"  Expected: {expected_ids}")
    print(f"  Found:    {found_ids}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_3_opposition() -> bool:
    print("--- Test 3: Opposition Detection (2nd harmonic / 180 deg) ---")
    field = ResonanceField()

    for i in range(12):
        deg = i * 30.0
        packet = WavePacket(f"pos_{deg:.0f}").with_attr("angle", Phase.from_degrees(deg))
        field.add(packet)

    target = Phase.from_degrees(0.0)
    results = field.query_harmonic("angle", target, 2, 0.95)

    print("  Target: 0 deg, Query: 2nd harmonic, Threshold: 0.95")
    print("  Matches:")
    for p, c in results:
        print(f"    {p.id} -> harmonic_coherence = {c:.4f}")

    expected_ids = ["pos_0", "pos_180"]
    found_ids = [p.id for p, _ in results]
    passed = len(found_ids) == 2 and all(e in found_ids for e in expected_ids)

    print(f"  Expected: {expected_ids}")
    print(f"  Found:    {found_ids}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_4_fuzzy_matching() -> bool:
    print("--- Test 4: Fuzzy Matching with Tolerance ---")
    field = ResonanceField()

    angles = [118.0, 120.0, 125.0, 130.0, 90.0, 0.0]
    for deg in angles:
        packet = WavePacket(f"pos_{deg:.0f}").with_attr("angle", Phase.from_degrees(deg))
        field.add(packet)

    target = Phase.from_degrees(0.0)
    orb = 8.0
    target_angle = 120.0

    results = field.query_fuzzy("angle", target, target_angle, orb)

    print(f"  Target: 0 deg, Looking for: 120 deg +/- 8 deg orb")
    print("  Matches:")
    for p, score in results:
        phase = p.phase_for("angle")
        print(f"    {p.id} -> distance from target = {phase.distance_degrees(target):.1f} deg, fuzzy score = {score:.4f}")

    score_120 = next((s for p, s in results if p.id == "pos_120"), None)
    score_118 = next((s for p, s in results if p.id == "pos_118"), None)
    score_125 = next((s for p, s in results if p.id == "pos_125"), None)
    score_130 = next((s for p, s in results if p.id == "pos_130"), None)

    print(f"  Scores: 120 deg={score_120:.4f}, 118 deg={score_118:.4f}, 125 deg={score_125:.4f}, 130 deg={score_130}")

    passed = (score_120 is not None and score_120 > 0.99
              and score_118 is not None and score_118 > 0.9
              and score_125 is not None and score_125 > 0.5
              and score_120 > score_118
              and score_118 > score_125
              and score_130 is None)

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_5_multi_attribute() -> bool:
    print("--- Test 5: Multi-Attribute Coherence ---")
    field = ResonanceField()

    field.add(WavePacket("A")
              .with_attr("vendor", Phase.from_degrees(30.0))
              .with_attr("category", Phase.from_degrees(120.0)))
    field.add(WavePacket("B")
              .with_attr("vendor", Phase.from_degrees(30.0))
              .with_attr("category", Phase.from_degrees(240.0)))
    field.add(WavePacket("C")
              .with_attr("vendor", Phase.from_degrees(30.0))
              .with_attr("category", Phase.from_degrees(120.0)))
    field.add(WavePacket("D")
              .with_attr("vendor", Phase.from_degrees(200.0))
              .with_attr("category", Phase.from_degrees(120.0)))

    target_vendor = Phase.from_degrees(30.0)
    target_category = Phase.from_degrees(120.0)

    print("  Target: vendor=30 deg, category=120 deg")
    print("  Combined coherence (product of per-attribute coherence):")

    scores = {}
    for p in field.packets:
        vc = p.phase_for("vendor").coherence(target_vendor) if p.phase_for("vendor") else 0.0
        cc = p.phase_for("category").coherence(target_category) if p.phase_for("category") else 0.0
        combined = vc * cc
        print(f"    {p.id} -> vendor={vc:.4f}, category={cc:.4f}, combined={combined:.4f}")
        scores[p.id] = combined

    passed = (scores["C"] > 0.99
              and scores["A"] > 0.99
              and scores["C"] > scores["B"]
              and scores["C"] > scores["D"])

    print(f"  C > B: {scores['C']} > {scores['B']} = {scores['C'] > scores['B']}")
    print(f"  C > D: {scores['C']} > {scores['D']} = {scores['C'] > scores['D']}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed
