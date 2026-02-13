"""Tests 14-16: Harmonic orthogonality, wraparound, scale resolution."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from wave import Phase, WavePacket
from field import ResonanceField


def test_14_harmonic_orthogonality() -> bool:
    print("--- Test 14: Harmonic Orthogonality ---")

    field = ResonanceField()
    test_angles = [0.0, 60.0, 72.0, 90.0, 120.0, 180.0, 240.0, 270.0, 288.0, 300.0]
    for deg in test_angles:
        field.add(WavePacket(f"pos_{deg:.0f}").with_attr("angle", Phase.from_degrees(deg)))

    target = Phase.from_degrees(0.0)
    threshold = 0.95

    h3 = field.query_harmonic("angle", target, 3, threshold)
    h3_ids = [p.id for p, _ in h3]

    h4 = field.query_harmonic("angle", target, 4, threshold)
    h4_ids = [p.id for p, _ in h4]

    h6 = field.query_harmonic("angle", target, 6, threshold)
    h6_ids = [p.id for p, _ in h6]

    h5 = field.query_harmonic("angle", target, 5, threshold)
    h5_ids = [p.id for p, _ in h5]

    print(f"  n=3 (120 deg): {h3_ids}")
    print(f"  n=4 (90 deg):  {h4_ids}")
    print(f"  n=5 (72 deg):  {h5_ids}")
    print(f"  n=6 (60 deg):  {h6_ids}")

    h3_no_90 = "pos_90" not in h3_ids and "pos_270" not in h3_ids
    h3_no_60 = "pos_60" not in h3_ids and "pos_300" not in h3_ids
    h4_no_120 = "pos_120" not in h4_ids and "pos_240" not in h4_ids
    h4_no_60 = "pos_60" not in h4_ids and "pos_300" not in h4_ids

    h3_has_120 = "pos_120" in h3_ids and "pos_240" in h3_ids
    h4_has_90 = "pos_90" in h4_ids and "pos_270" in h4_ids
    h5_has_72 = "pos_72" in h5_ids and "pos_288" in h5_ids

    print("\n  Cross-talk checks:")
    print(f"    n=3 excludes 90 deg/270 deg:  {h3_no_90}")
    print(f"    n=3 excludes 60 deg/300 deg:  {h3_no_60}")
    print(f"    n=4 excludes 120 deg/240 deg: {h4_no_120}")
    print(f"    n=4 excludes 60 deg/300 deg:  {h4_no_60}")
    print("  Inclusion checks:")
    print(f"    n=3 finds 120 deg/240 deg: {h3_has_120}")
    print(f"    n=4 finds 90 deg/270 deg:  {h4_has_90}")
    print(f"    n=5 finds 72 deg/288 deg:  {h5_has_72}")

    passed = (h3_no_90 and h3_no_60 and h4_no_120 and h4_no_60
              and h3_has_120 and h4_has_90 and h5_has_72)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_15_wraparound() -> bool:
    print("--- Test 15: Phase Wraparound at 0 deg/360 deg Boundary ---")

    a = Phase.from_degrees(1.0)
    b = Phase.from_degrees(359.0)
    dist = a.distance_degrees(b)

    print(f"  1 deg to 359 deg: distance = {dist:.4f} deg (expected: 2.0 deg)")
    dist_ok = abs(dist - 2.0) < 0.001

    coh = a.coherence(b)
    print(f"  1 deg to 359 deg: coherence = {coh:.6f} (expected: ~0.9994)")
    coh_ok = coh > 0.999

    target = Phase.from_degrees(0.0)
    score_1 = target.fuzzy_match(a, 0.0, 8.0)
    score_359 = target.fuzzy_match(b, 0.0, 8.0)
    print(f"  Fuzzy(0 deg -> 1 deg, target=0 deg, orb=8 deg):   {score_1:.6f}")
    print(f"  Fuzzy(0 deg -> 359 deg, target=0 deg, orb=8 deg):  {score_359:.6f}")
    fuzzy_ok = score_1 > 0.97 and score_359 > 0.97 and abs(score_1 - score_359) < 0.0001

    dir_1_to_359 = a.directed_distance_degrees(b)
    dir_359_to_1 = b.directed_distance_degrees(a)
    print(f"  Directed 1 deg -> 359 deg: {dir_1_to_359:.4f} deg (expected: 358.0 deg)")
    print(f"  Directed 359 deg -> 1 deg: {dir_359_to_1:.4f} deg (expected: 2.0 deg)")
    dir_ok = abs(dir_1_to_359 - 358.0) < 0.001 and abs(dir_359_to_1 - 2.0) < 0.001

    field = ResonanceField()
    for deg in [357.0, 358.0, 359.0, 0.0, 1.0, 2.0, 3.0, 180.0]:
        field.add(WavePacket(f"pos_{deg:.0f}").with_attr("angle", Phase.from_degrees(deg)))

    near_zero = field.query_fuzzy("angle", target, 0.0, 5.0)
    near_ids = [p.id for p, _ in near_zero]
    print(f"  Entities within 5 deg of 0 deg: {near_ids}")
    field_ok = (
        "pos_0" in near_ids and "pos_1" in near_ids
        and "pos_2" in near_ids and "pos_3" in near_ids
        and "pos_357" in near_ids and "pos_358" in near_ids
        and "pos_359" in near_ids and "pos_180" not in near_ids
    )

    passed = dist_ok and coh_ok and fuzzy_ok and dir_ok and field_ok
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_16_scale_resolution() -> bool:
    print("--- Test 16: Scale Resolution (360 Values) ---")

    bucket_count = 360
    field = ResonanceField()

    for i in range(360):
        field.add(WavePacket(f"val_{i}").with_attr("v", Phase.from_value(i, bucket_count)))

    bucket_angle_cos = math.cos(2.0 * math.pi / bucket_count)
    threshold = (1.0 + bucket_angle_cos) / 2.0
    print(f"  cos(1 deg) = {bucket_angle_cos:.6f}, threshold = {threshold:.6f}")

    perfect = True
    total_matches = 0
    false_positives = 0

    for i in range(360):
        target = Phase.from_value(i, bucket_count)
        results = field.query_exact("v", target, threshold)
        total_matches += len(results)
        if len(results) != 1:
            if i < 5 or i > 355:
                print(f"  val_{i}: got {len(results)} matches (expected 1)")
            perfect = False
            false_positives += max(0, len(results) - 1)
        elif results[0][0].id != f"val_{i}":
            print(f"  val_{i}: wrong match! got {results[0][0].id}")
            perfect = False
            false_positives += 1

    print(f"  360 queries, {total_matches} total matches, {false_positives} false positives")
    print(f"  Perfect resolution (each query -> exactly 1 correct match): {perfect}")

    target_0 = Phase.from_value(0, bucket_count)
    h3_results = field.query_harmonic("v", target_0, 3, 0.99)
    h3_ids = [p.id for p, _ in h3_results]
    h3_has_centers = "val_0" in h3_ids and "val_120" in h3_ids and "val_240" in h3_ids
    print(f"  Harmonic n=3 from val_0: {len(h3_results)} matches (5 per group x 3 groups), centers present: {h3_has_centers}")

    h4_results = field.query_harmonic("v", target_0, 4, 0.99)
    h4_ids = [p.id for p, _ in h4_results]
    h4_has_centers = ("val_0" in h4_ids and "val_90" in h4_ids
                      and "val_180" in h4_ids and "val_270" in h4_ids)
    print(f"  Harmonic n=4 from val_0: {len(h4_results)} matches (5 per group x 4 groups), centers present: {h4_has_centers}")

    h3_nyquist = math.cos(3.0 * 2.0 * math.pi / bucket_count)
    h4_nyquist = math.cos(4.0 * 2.0 * math.pi / bucket_count)
    print(f"  Nyquist floor at n=3: cos(3 deg) = {h3_nyquist:.6f} (threshold must exceed this for single-value precision)")
    print(f"  Nyquist floor at n=4: cos(4 deg) = {h4_nyquist:.6f}")

    passed = (perfect and h3_has_centers and len(h3_results) == 15
              and h4_has_centers and len(h4_results) == 20)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed
