"""Tests 18-20: Bucket indexing, multi-attribute torus, dynamic mutation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wave import Phase, WavePacket
from field import ResonanceField, BucketIndex, MultiAttrBucketIndex


def test_18_bucket_index() -> bool:
    print("--- Test 18: Direct Position Lookup (Placement as Indexing) ---")
    print()

    n_entities = 1000
    bucket_count = 360
    golden_angle = 137.50776405003785

    field = ResonanceField()
    index = BucketIndex("pos", bucket_count)

    for i in range(n_entities):
        deg = (i * golden_angle) % 360.0
        packet = WavePacket(f"e_{i}").with_attr("pos", Phase.from_degrees(deg))
        field.add(packet)
        index.insert(packet)

    print(f"  Entities: {n_entities}, Buckets: {bucket_count}")
    print(f"  Density: {n_entities / bucket_count:.2f} entities/bucket average")
    print()

    all_correct = True

    # === Part 1: Exact match queries at various thresholds ===
    print("  [Exact Match Queries]")
    thresholds = [0.95, 0.99, 0.999]
    for threshold in thresholds:
        target = Phase.from_degrees(45.0)

        full_results = field.query_exact("pos", target, threshold)
        idx_results, examined = index.query_exact(target, threshold)

        full_ids = sorted(p.id for p, _ in full_results)
        idx_ids = sorted(p.id for p, _ in idx_results)

        correct = full_ids == idx_ids
        if not correct:
            all_correct = False

        selectivity = examined / n_entities * 100.0
        print(f"    threshold={threshold:.3f}: found={len(idx_results)}, examined={examined}/{n_entities} "
              f"({selectivity:.1f}%), correct={'YES' if correct else 'NO'}")

    # === Part 2: Harmonic queries ===
    print()
    print("  [Harmonic Queries (threshold=0.90)]")
    harmonics = [2, 3, 4, 6, 12]
    harm_threshold = 0.9
    for h in harmonics:
        target = Phase.from_degrees(0.0)

        full_results = field.query_harmonic("pos", target, h, harm_threshold)
        idx_results, examined = index.query_harmonic(target, h, harm_threshold)

        full_ids = sorted(p.id for p, _ in full_results)
        idx_ids = sorted(p.id for p, _ in idx_results)

        correct = full_ids == idx_ids
        if not correct:
            all_correct = False

        selectivity = examined / n_entities * 100.0
        print(f"    harmonic n={h:>2}: found={len(idx_results):>3}, examined={examined:>3}/{n_entities} "
              f"({selectivity:>5.1f}%), correct={'YES' if correct else 'NO'}")

    # === Part 3: Multi-target verification ===
    print()
    print("  [Multi-Target Verification]")
    targets_deg = [0.0, 30.0, 90.0, 137.5, 200.0, 315.0]
    multi_correct = 0
    multi_total = 0
    total_examined = 0
    total_possible = 0

    for deg in targets_deg:
        target = Phase.from_degrees(deg)

        full = field.query_exact("pos", target, 0.95)
        indexed, ex = index.query_exact(target, 0.95)
        f_ids = sorted(p.id for p, _ in full)
        i_ids = sorted(p.id for p, _ in indexed)
        multi_total += 1
        total_examined += ex
        total_possible += n_entities
        if f_ids == i_ids:
            multi_correct += 1

        full = field.query_harmonic("pos", target, 3, 0.9)
        indexed, ex = index.query_harmonic(target, 3, 0.9)
        f_ids = sorted(p.id for p, _ in full)
        i_ids = sorted(p.id for p, _ in indexed)
        multi_total += 1
        total_examined += ex
        total_possible += n_entities
        if f_ids == i_ids:
            multi_correct += 1

    avg_selectivity = total_examined / total_possible * 100.0
    print(f"    {multi_correct} of {multi_total} queries match full scan")
    print(f"    Average selectivity: {avg_selectivity:.1f}% of entities examined")
    if multi_correct != multi_total:
        all_correct = False

    # === Summary ===
    print()
    print("  Key insight: placement on the circle IS the index.")
    print("  No separate index structure to build or maintain.")
    print("  Insert = O(1), Exact query = O(spread * density),")
    print("  Harmonic query = O(n * spread * density)")

    passed = all_correct
    print()
    print(f"  All indexed queries match full scan: {all_correct}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_19_multi_attr_index() -> bool:
    print("--- Test 19: Multi-Attribute Torus Index (2D Compound Queries) ---")
    print()

    n_entities = 500
    bucket_count = 60
    golden_angle = 137.50776405003785
    silver_angle = 222.49223594996215  # 360 - golden

    field = ResonanceField()
    index = MultiAttrBucketIndex("x", "y", bucket_count)

    for i in range(n_entities):
        deg_x = (i * golden_angle) % 360.0
        deg_y = (i * silver_angle) % 360.0
        packet = (WavePacket(f"e_{i}")
                  .with_attr("x", Phase.from_degrees(deg_x))
                  .with_attr("y", Phase.from_degrees(deg_y)))
        field.add(packet)
        index.insert(packet)

    grid_cells = bucket_count * bucket_count
    print(f"  Entities: {n_entities}, Buckets per dim: {bucket_count}")
    print(f"  Grid cells: {grid_cells} ({bucket_count}x{bucket_count})")
    print(f"  Density: {n_entities / grid_cells:.4f} entities/cell average")
    print()

    all_correct = True

    # === Part 1: Exact+Exact compound queries ===
    print("  [Exact+Exact Compound Queries (threshold=0.95)]")
    targets = [(45.0, 100.0), (180.0, 270.0), (0.0, 0.0), (200.0, 50.0)]
    threshold = 0.95

    for dx, dy in targets:
        target_x = Phase.from_degrees(dx)
        target_y = Phase.from_degrees(dy)

        # Ground truth: full scan
        full = []
        for p in field.packets:
            cx = p.phase_for("x").coherence(target_x) if p.phase_for("x") else -1.0
            cy = p.phase_for("y").coherence(target_y) if p.phase_for("y") else -1.0
            if cx >= threshold and cy >= threshold:
                full.append((p, cx * cy))

        idx_results, examined = index.query_exact_both(target_x, target_y, threshold)

        full_ids = sorted(p.id for p, _ in full)
        idx_ids = sorted(p.id for p, _ in idx_results)

        correct = full_ids == idx_ids
        if not correct:
            all_correct = False

        selectivity = examined / n_entities * 100.0
        print(f"    target=({dx:.0f} deg,{dy:.0f} deg): found={len(idx_results)}, examined={examined}/{n_entities} "
              f"({selectivity:.1f}%), correct={'YES' if correct else 'NO'}")

    # === Part 2: Exact+Harmonic compound queries ===
    print()
    print("  [Exact+Harmonic Compound Queries (exact x @ 0.95, harmonic n=3 y @ 0.85)]")
    targets_2 = [(0.0, 0.0), (120.0, 60.0), (270.0, 180.0)]
    threshold_a = 0.95
    threshold_b = 0.85
    harmonic = 3

    for dx, dy in targets_2:
        target_x = Phase.from_degrees(dx)
        target_y = Phase.from_degrees(dy)

        full = []
        for p in field.packets:
            cx = p.phase_for("x").coherence(target_x) if p.phase_for("x") else -1.0
            cy = p.phase_for("y").harmonic_coherence(target_y, harmonic) if p.phase_for("y") else -1.0
            if cx >= threshold_a and cy >= threshold_b:
                full.append((p, cx * cy))

        idx_results, examined = index.query_exact_harmonic(
            target_x, threshold_a, target_y, harmonic, threshold_b
        )

        full_ids = sorted(p.id for p, _ in full)
        idx_ids = sorted(p.id for p, _ in idx_results)

        correct = full_ids == idx_ids
        if not correct:
            all_correct = False

        selectivity = examined / n_entities * 100.0
        print(f"    target=({dx:.0f} deg,{dy:.0f} deg) n={harmonic}: found={len(idx_results)}, examined={examined}/{n_entities} "
              f"({selectivity:.1f}%), correct={'YES' if correct else 'NO'}")

    # === Part 3: 2D vs 1D selectivity comparison ===
    print()
    print("  [2D vs 1D Selectivity Comparison]")

    idx_1d = BucketIndex("x", bucket_count)
    for p in field.packets:
        idx_1d.insert(p)

    target_x = Phase.from_degrees(45.0)
    target_y = Phase.from_degrees(100.0)

    _, examined_1d = idx_1d.query_exact(target_x, threshold)
    _, examined_2d = index.query_exact_both(target_x, target_y, threshold)

    sel_1d = examined_1d / n_entities * 100.0
    sel_2d = examined_2d / n_entities * 100.0
    improvement = examined_1d / examined_2d if examined_2d > 0 else float("inf")

    print(f"    1D index (x only):  examined={examined_1d}/{n_entities} ({sel_1d:.1f}%)")
    print(f"    2D index (x and y): examined={examined_2d}/{n_entities} ({sel_2d:.1f}%)")
    print(f"    Improvement factor: {improvement:.1f}x fewer entities examined")

    print()
    print("  Key insight: each indexed dimension narrows independently.")
    print("  2D torus selectivity ~ (1D selectivity)^2 — multiplicative, not additive.")

    passed = all_correct
    print()
    print(f"  All compound queries match full scan: {all_correct}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_20_dynamic_mutation() -> bool:
    print("--- Test 20: Dynamic Mutation (Insert / Remove / Update) ---")
    print()

    n_initial = 200
    bucket_count = 60
    golden_angle = 137.50776405003785

    index = BucketIndex("pos", bucket_count)
    truth: list[WavePacket | None] = []

    for i in range(n_initial):
        deg = (i * golden_angle) % 360.0
        packet = WavePacket(f"e_{i}").with_attr("pos", Phase.from_degrees(deg))
        index.insert(packet)
        truth.append(packet)

    print(f"  Initial entities: {n_initial}, Buckets: {bucket_count}")
    print(f"  Active: {index.active_count()}")
    print()

    threshold = 0.95
    all_correct = True

    def truth_query_exact(target: Phase, threshold: float) -> list[str]:
        ids = []
        for p in truth:
            if p is not None:
                ph = p.phase_for("pos")
                if ph is not None and ph.coherence(target) >= threshold:
                    ids.append(p.id)
        return sorted(ids)

    def truth_query_harmonic(target: Phase, n: int, threshold: float) -> list[str]:
        ids = []
        for p in truth:
            if p is not None:
                ph = p.phase_for("pos")
                if ph is not None and ph.harmonic_coherence(target, n) >= threshold:
                    ids.append(p.id)
        return sorted(ids)

    # === Phase 1: Remove 50 entities ===
    print("  [Phase 1: Remove 50 entities]")
    remove_ids = [f"e_{i * 4}" for i in range(50)]

    for id_ in remove_ids:
        removed = index.remove_by_id(id_)
        if not removed:
            all_correct = False
        # Mirror in ground truth
        for pos, opt in enumerate(truth):
            if opt is not None and opt.id == id_:
                truth[pos] = None
                break

    active_after_remove = index.active_count()
    print(f"    Removed: {len(remove_ids)}, Active: {active_after_remove}")

    double_remove = index.remove_by_id("e_0")
    double_ok = not double_remove
    if not double_ok:
        all_correct = False
    print(f"    Double-remove returns false: {'YES' if double_ok else 'NO'}")

    target = Phase.from_degrees(45.0)
    idx_results, _ = index.query_exact(target, threshold)
    idx_ids = sorted(p.id for p, _ in idx_results)
    truth_ids = truth_query_exact(target, threshold)

    remove_correct = idx_ids == truth_ids
    if not remove_correct:
        all_correct = False
    print(f"    Query after remove: found={len(idx_results)}, correct={'YES' if remove_correct else 'NO'}")

    # === Phase 2: Insert 30 new entities ===
    print()
    print("  [Phase 2: Insert 30 new entities]")

    for i in range(30):
        deg = (i * 12.345 + 77.0) % 360.0
        packet = WavePacket(f"new_{i}").with_attr("pos", Phase.from_degrees(deg))
        index.insert(packet)
        truth.append(packet)

    active_after_insert = index.active_count()
    print(f"    Inserted: 30, Active: {active_after_insert}")

    idx_results, _ = index.query_exact(target, threshold)
    idx_ids = sorted(p.id for p, _ in idx_results)
    truth_ids = truth_query_exact(target, threshold)

    insert_correct = idx_ids == truth_ids
    if not insert_correct:
        all_correct = False
    print(f"    Query after insert: found={len(idx_results)}, correct={'YES' if insert_correct else 'NO'}")

    # === Phase 3: Update 20 entities (reposition) ===
    print()
    print("  [Phase 3: Update 20 entities (reposition)]")

    update_ids = [f"e_{i * 2 + 1}" for i in range(20)]
    updates_applied = 0

    for i, id_ in enumerate(update_ids):
        new_deg = (i * 18.0 + 5.0) % 360.0
        new_packet = WavePacket(id_).with_attr("pos", Phase.from_degrees(new_deg))

        found = index.update(id_, new_packet)
        if found:
            updates_applied += 1

        # Mirror in ground truth: remove old, add new
        for pos, opt in enumerate(truth):
            if opt is not None and opt.id == id_:
                truth[pos] = None
                break
        truth.append(new_packet)

    active_after_update = index.active_count()
    print(f"    Update attempts: {len(update_ids)}, Applied: {updates_applied}, Active: {active_after_update}")

    test_targets = [0.0, 45.0, 90.0, 180.0, 270.0]
    multi_correct = 0

    for deg in test_targets:
        t = Phase.from_degrees(deg)
        idx_results, _ = index.query_exact(t, threshold)
        idx_ids = sorted(p.id for p, _ in idx_results)
        t_ids = truth_query_exact(t, threshold)
        if idx_ids == t_ids:
            multi_correct += 1
        else:
            all_correct = False

    print(f"    Post-update exact queries: {multi_correct}/{len(test_targets)} correct")

    # === Phase 4: Harmonic queries after all mutations ===
    print()
    print("  [Phase 4: Harmonic queries after all mutations]")

    harm_targets = [0.0, 120.0, 240.0]
    harm_threshold = 0.85
    harm_correct = 0

    for deg in harm_targets:
        t = Phase.from_degrees(deg)
        idx_results, _ = index.query_harmonic(t, 3, harm_threshold)
        idx_ids = sorted(p.id for p, _ in idx_results)
        t_ids = truth_query_harmonic(t, 3, harm_threshold)
        if idx_ids == t_ids:
            harm_correct += 1
        else:
            all_correct = False

    print(f"    Harmonic n=3 post-mutation: {harm_correct}/{len(harm_targets)} correct")

    # === Summary ===
    print()
    print("  Mutation summary:")
    print(f"    {n_initial} initial - {len(remove_ids)} removed + 30 inserted - {updates_applied} repositioned = {active_after_update} active")
    print("  Key insight: mutations are local operations on the circle.")
    print("  Remove = tombstone + bucket cleanup. Update = remove + re-insert.")
    print("  No global rebuild needed. Queries remain correct throughout.")

    passed = all_correct
    print()
    print(f"  All queries correct after mutations: {all_correct}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed
