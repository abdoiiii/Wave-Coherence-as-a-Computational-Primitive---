"""Test 17: Density scaling and capacity limits."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from wave import Phase, WavePacket
from field import ResonanceField


def test_17_scaling_limits() -> bool:
    print("--- Test 17: Density Scaling and Capacity Limits ---")
    print()

    scenarios = [
        ("7 in 12", 7, 12),
        ("9 in 27", 9, 27),
        ("12 in 12 (saturated)", 12, 12),
        ("20 in 60", 20, 60),
        ("50 in 360", 50, 360),
        ("100 in 360", 100, 360),
        ("200 in 360", 200, 360),
        ("360 in 360 (saturated)", 360, 360),
    ]

    results = []

    for name, n_objects, buckets in scenarios:
        golden_angle = 137.50776405003785
        positions_deg = [(i * golden_angle) % 360.0 for i in range(n_objects)]

        field = ResonanceField()
        for i, deg in enumerate(positions_deg):
            field.add(WavePacket(f"obj_{i}").with_attr("pos", Phase.from_degrees(deg)))

        # 1. Find minimum angular separation between any pair
        min_sep = 360.0
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                pi = Phase.from_degrees(positions_deg[i])
                pj = Phase.from_degrees(positions_deg[j])
                sep = pi.distance_degrees(pj)
                if sep < min_sep:
                    min_sep = sep

        # 2. Compute maximum harmonic needed to resolve closest pair
        threshold = 0.9
        if min_sep > 0.001:
            max_harmonic = math.ceil(math.degrees(math.acos(threshold)) / min_sep)
        else:
            max_harmonic = 9999

        # 3. Test exact match: can each object be uniquely identified?
        bucket_angle_cos = math.cos(2.0 * math.pi / buckets)
        exact_threshold = (1.0 + bucket_angle_cos) / 2.0

        exact_ok = True
        for i, deg in enumerate(positions_deg):
            target = Phase.from_degrees(deg)
            matches = field.query_exact("pos", target, exact_threshold)
            found_self = any(p.id == f"obj_{i}" for p, _ in matches)
            if not found_self or len(matches) != 1:
                exact_ok = False

        # 4. Test triadic detection
        target_phase = Phase.from_degrees(positions_deg[0])
        triadic_results = field.query_harmonic("pos", target_phase, 3, 0.85)

        noise_triadic = 0
        for p, _ in triadic_results:
            p_phase = p.phase_for("pos")
            dist = target_phase.distance_degrees(p_phase)
            near_0 = dist < 10.0
            near_120 = abs(dist - 120.0) < 10.0
            near_240 = abs(dist - 240.0) < 10.0
            if not (near_0 or near_120 or near_240):
                noise_triadic += 1
        triadic_clean = noise_triadic == 0

        density = n_objects / buckets * 100.0

        results.append({
            "name": name,
            "n_objects": n_objects,
            "buckets": buckets,
            "density_pct": density,
            "min_separation_deg": min_sep,
            "max_harmonic_needed": max_harmonic,
            "exact_match_ok": exact_ok,
            "triadic_clean": triadic_clean,
        })

    # Print results table
    print(f"  {'Configuration':<30} {'N':>4} {'B':>5} {'N/B %':>7} {'Min Sep':>9} {'Max n':>8} {'Exact':>7} {'Triad':>7}")
    print(f"  {'-------------':<30} {'---':>4} {'---':>5} {'-----':>7} {'-------':>9} {'-----':>8} {'-----':>7} {'-----':>7}")

    for r in results:
        print(f"  {r['name']:<30} {r['n_objects']:>4} {r['buckets']:>5} {r['density_pct']:>6.1f}% "
              f"{r['min_separation_deg']:>8.3f} deg {r['max_harmonic_needed']:>8} "
              f"{'OK' if r['exact_match_ok'] else 'FAIL':>7} "
              f"{'clean' if r['triadic_clean'] else 'noisy':>7}")

    print()
    first_exact_fail = next((r for r in results if not r["exact_match_ok"]), None)
    first_triadic_noise = next((r for r in results if not r["triadic_clean"]), None)

    if first_exact_fail:
        print(f"  Exact match degrades at: {first_exact_fail['name']} ({first_exact_fail['density_pct']:.1f}% density)")
    else:
        print("  Exact match: no degradation across all scenarios")

    if first_triadic_noise:
        print(f"  Triadic (n=3) noise begins at: {first_triadic_noise['name']} ({first_triadic_noise['density_pct']:.1f}% density)")
    else:
        print("  Triadic detection: clean across all scenarios")

    # Collision probability
    print()
    print("  Bucket collision probability (birthday problem approximation):")
    for r in results:
        n = r["n_objects"]
        b = r["buckets"]
        p_collision = 1.0 - math.exp(-n * (n - 1) / (2.0 * b))
        print(f"    {r['name']}: P(collision) = {p_collision * 100.0:.1f}%")

    # Design rules
    print()
    print("  Design rules:")
    print("    - Exact match requires density < 100% (no two objects in same bucket)")
    if first_triadic_noise:
        print(f"    - Clean triadic detection requires density < ~{first_triadic_noise['density_pct']:.0f}% at threshold 0.85")
    print("    - Resolution harmonic scales inversely with minimum separation")
    print("    - Collision probability follows birthday problem: P ~ 1 - e^(-N^2/2B)")

    # Pass conditions
    smallest_ok = results[0]["exact_match_ok"] and results[0]["triadic_clean"]
    degrades = any(not r["exact_match_ok"] or not r["triadic_clean"] for r in results)
    harmonic_scales = results[0]["max_harmonic_needed"] < results[-1]["max_harmonic_needed"]
    exact_only_at_saturation = all(r["exact_match_ok"] or r["density_pct"] >= 100.0 for r in results)

    passed = smallest_ok and degrades and harmonic_scales and exact_only_at_saturation

    print()
    print(f"  Smallest config fully clean: {smallest_ok}")
    print(f"  System degrades at higher density: {degrades}")
    print(f"  Resolution harmonic increases with density: {harmonic_scales}")
    print(f"  Exact match only fails at 100% saturation: {exact_only_at_saturation}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed
