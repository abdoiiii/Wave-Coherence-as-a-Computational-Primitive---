"""Tests 8-9: Wave vs linear scan, harmonic vs JOIN."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wave import Phase, WavePacket
from field import ResonanceField


def test_8_wave_vs_linear() -> bool:
    print("--- Test 8: Wave vs Linear Scan (Correctness) ---")

    bucket_count = 100
    entity_count = 1000
    target_value = 42

    field = ResonanceField()
    values = []

    for i in range(entity_count):
        val = (i * 37 + 13) % bucket_count
        id_ = f"entity_{i}"
        values.append((id_, val))
        packet = WavePacket(id_).with_attr("value", Phase.from_value(val, bucket_count))
        field.add(packet)

    # Method A: Linear scan with value comparison
    linear_results = sorted([id_ for id_, v in values if v == target_value])

    # Method B: Coherence scan
    target_phase = Phase.from_value(target_value, bucket_count)
    wave_results = sorted([p.id for p, _ in field.query_exact("value", target_phase, 0.9999)])

    print(f"  {entity_count} entities, bucket_count={bucket_count}, target_value={target_value}")
    print(f"  Linear scan found: {len(linear_results)} matches")
    print(f"  Wave scan found:   {len(wave_results)} matches")

    passed = linear_results == wave_results

    if not passed:
        print("  MISMATCH!")
        print(f"  Linear: {linear_results[:10]}")
        print(f"  Wave:   {wave_results[:10]}")
    else:
        print("  Result sets are IDENTICAL")

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_9_harmonic_vs_join() -> bool:
    print("--- Test 9: Harmonic Query vs JOIN ---")

    field = ResonanceField()

    group_centers = [0.0, 90.0, 120.0, 240.0]
    group_names = ["group_0", "group_90", "group_120", "group_240"]

    for gi, center in enumerate(group_centers):
        for j in range(25):
            offset = (j - 12.0) * 0.2
            deg = center + offset
            id_ = f"{group_names[gi]}_{j}"
            packet = WavePacket(id_).with_attr("pos", Phase.from_degrees(deg))
            field.add(packet)

    target = Phase.from_degrees(3.0)
    harmonic_results = field.query_harmonic("pos", target, 3, 0.85)

    group_counts = [0, 0, 0, 0]
    for p, _ in harmonic_results:
        for gi, name in enumerate(group_names):
            if p.id.startswith(name):
                group_counts[gi] += 1

    print("  Target: entity at 3 deg (in group_0)")
    print("  3rd harmonic scan results by group:")
    for gi, name in enumerate(group_names):
        print(f"    {name}: {group_counts[gi]} entities found")
    print(f"  Total harmonic matches: {len(harmonic_results)}")

    found_120 = group_counts[2] > 0
    found_240 = group_counts[3] > 0
    not_found_90 = group_counts[1] == 0

    print(f"  Found group_120: {found_120}")
    print(f"  Found group_240: {found_240}")
    print(f"  Excluded group_90: {not_found_90}")
    print("  Key insight: Single harmonic scan found BOTH 120 deg and 240 deg groups")
    print("  Traditional approach would need 2 JOINs for the same result")

    passed = found_120 and found_240 and not_found_90
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed
