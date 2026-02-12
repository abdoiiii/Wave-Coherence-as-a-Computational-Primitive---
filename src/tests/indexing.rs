use crate::wave::{Phase, WavePacket};
use crate::field::{ResonanceField, BucketIndex, MultiAttrBucketIndex};

// =============================================================================
// Test 18: Direct Position Lookup — Placement as Indexing
// =============================================================================
pub fn test_18_bucket_index() -> bool {
    println!("--- Test 18: Direct Position Lookup (Placement as Indexing) ---");
    println!();

    // Hypothesis: the circle IS the index. Placing an entity at a position on
    // the circle automatically indexes it. Queries compute target bucket(s) and
    // check only nearby buckets, achieving sub-linear lookup without maintaining
    // a separate index structure.

    let n_entities = 1000usize;
    let bucket_count = 360u32;
    let golden_angle = 137.50776405003785_f64;

    // Place entities using golden angle for irrational spacing
    let mut field = ResonanceField::new();
    let mut index = BucketIndex::new("pos", bucket_count);

    for i in 0..n_entities {
        let deg = (i as f64 * golden_angle) % 360.0;
        let packet = WavePacket::new(&format!("e_{i}"))
            .with_attr("pos", Phase::from_degrees(deg));
        field.add(packet.clone());
        index.insert(packet);
    }

    println!("  Entities: {n_entities}, Buckets: {bucket_count}");
    println!("  Density: {:.2} entities/bucket average",
        n_entities as f64 / bucket_count as f64);
    println!();

    let mut all_correct = true;

    // === Part 1: Exact match queries at various thresholds ===
    println!("  [Exact Match Queries]");
    let thresholds = [0.95, 0.99, 0.999];
    for &threshold in &thresholds {
        let target = Phase::from_degrees(45.0);

        // Full scan (ground truth)
        let full_results = field.query_exact("pos", &target, threshold);

        // Indexed query
        let (idx_results, examined) = index.query_exact(&target, threshold);

        // Compare: same set of packet IDs
        let mut full_ids: Vec<&str> = full_results.iter()
            .map(|(p, _)| p.id.as_str()).collect();
        let mut idx_ids: Vec<&str> = idx_results.iter()
            .map(|(p, _)| p.id.as_str()).collect();
        full_ids.sort();
        idx_ids.sort();

        let correct = full_ids == idx_ids;
        if !correct { all_correct = false; }

        let selectivity = examined as f64 / n_entities as f64 * 100.0;
        println!("    threshold={:.3}: found={}, examined={}/{} ({:.1}%), correct={}",
            threshold, idx_results.len(), examined, n_entities, selectivity,
            if correct { "YES" } else { "NO" });
    }

    // === Part 2: Harmonic queries ===
    println!();
    println!("  [Harmonic Queries (threshold=0.90)]");
    let harmonics = [2u32, 3, 4, 6, 12];
    let harm_threshold = 0.9;
    for &h in &harmonics {
        let target = Phase::from_degrees(0.0);

        let full_results = field.query_harmonic("pos", &target, h, harm_threshold);
        let (idx_results, examined) = index.query_harmonic(&target, h, harm_threshold);

        let mut full_ids: Vec<&str> = full_results.iter()
            .map(|(p, _)| p.id.as_str()).collect();
        let mut idx_ids: Vec<&str> = idx_results.iter()
            .map(|(p, _)| p.id.as_str()).collect();
        full_ids.sort();
        idx_ids.sort();

        let correct = full_ids == idx_ids;
        if !correct { all_correct = false; }

        let selectivity = examined as f64 / n_entities as f64 * 100.0;
        println!("    harmonic n={:>2}: found={:>3}, examined={:>3}/{} ({:>5.1}%), correct={}",
            h, idx_results.len(), examined, n_entities, selectivity,
            if correct { "YES" } else { "NO" });
    }

    // === Part 3: Multi-target verification across the circle ===
    println!();
    println!("  [Multi-Target Verification]");
    let targets_deg = [0.0, 30.0, 90.0, 137.5, 200.0, 315.0];
    let mut multi_correct = 0usize;
    let mut multi_total = 0usize;
    let mut total_examined = 0usize;
    let mut total_possible = 0usize;

    for &deg in &targets_deg {
        let target = Phase::from_degrees(deg);

        // Exact match at 0.95
        let full = field.query_exact("pos", &target, 0.95);
        let (indexed, ex) = index.query_exact(&target, 0.95);
        let mut f_ids: Vec<&str> = full.iter().map(|(p,_)| p.id.as_str()).collect();
        let mut i_ids: Vec<&str> = indexed.iter().map(|(p,_)| p.id.as_str()).collect();
        f_ids.sort(); i_ids.sort();
        multi_total += 1;
        total_examined += ex;
        total_possible += n_entities;
        if f_ids == i_ids { multi_correct += 1; }

        // Harmonic n=3 at 0.9
        let full = field.query_harmonic("pos", &target, 3, 0.9);
        let (indexed, ex) = index.query_harmonic(&target, 3, 0.9);
        let mut f_ids: Vec<&str> = full.iter().map(|(p,_)| p.id.as_str()).collect();
        let mut i_ids: Vec<&str> = indexed.iter().map(|(p,_)| p.id.as_str()).collect();
        f_ids.sort(); i_ids.sort();
        multi_total += 1;
        total_examined += ex;
        total_possible += n_entities;
        if f_ids == i_ids { multi_correct += 1; }
    }

    let avg_selectivity = total_examined as f64 / total_possible as f64 * 100.0;
    println!("    {} of {} queries match full scan", multi_correct, multi_total);
    println!("    Average selectivity: {:.1}% of entities examined", avg_selectivity);
    if multi_correct != multi_total { all_correct = false; }

    // === Summary ===
    println!();
    println!("  Key insight: placement on the circle IS the index.");
    println!("  No separate index structure to build or maintain.");
    println!("  Insert = O(1), Exact query = O(spread * density),");
    println!("  Harmonic query = O(n * spread * density)");

    let pass = all_correct;
    println!();
    println!("  All indexed queries match full scan: {}", all_correct);
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 19: Multi-Attribute Torus Index — 2D Compound Queries
// =============================================================================
pub fn test_19_multi_attr_index() -> bool {
    println!("--- Test 19: Multi-Attribute Torus Index (2D Compound Queries) ---");
    println!();

    // Hypothesis: multi-attribute indexing on a 2D torus (B×B grid) enables
    // compound queries (exact+exact, exact+harmonic) with selectivity that
    // improves multiplicatively over 1D indexing.

    let n_entities = 500usize;
    let bucket_count = 60u32;
    let golden_angle = 137.50776405003785_f64;
    let silver_angle = 222.49223594996215_f64; // 360 - golden (independent irrational)

    // Build flat field (ground truth) and 2D indexed field
    let mut field = ResonanceField::new();
    let mut index = MultiAttrBucketIndex::new("x", "y", bucket_count);

    for i in 0..n_entities {
        let deg_x = (i as f64 * golden_angle) % 360.0;
        let deg_y = (i as f64 * silver_angle) % 360.0;
        let packet = WavePacket::new(&format!("e_{i}"))
            .with_attr("x", Phase::from_degrees(deg_x))
            .with_attr("y", Phase::from_degrees(deg_y));
        field.add(packet.clone());
        index.insert(packet);
    }

    let grid_cells = (bucket_count as u64 * bucket_count as u64) as f64;
    println!("  Entities: {n_entities}, Buckets per dim: {bucket_count}");
    println!("  Grid cells: {} ({}x{})", bucket_count as u64 * bucket_count as u64, bucket_count, bucket_count);
    println!("  Density: {:.4} entities/cell average", n_entities as f64 / grid_cells);
    println!();

    let mut all_correct = true;

    // === Part 1: Exact+Exact compound queries ===
    println!("  [Exact+Exact Compound Queries (threshold=0.95)]");
    let targets = [(45.0, 100.0), (180.0, 270.0), (0.0, 0.0), (200.0, 50.0)];
    let threshold = 0.95;

    for &(dx, dy) in &targets {
        let target_x = Phase::from_degrees(dx);
        let target_y = Phase::from_degrees(dy);

        // Ground truth: full scan
        let full: Vec<(&WavePacket, f64)> = field.packets.iter()
            .filter_map(|p| {
                let cx = p.phase_for("x").map(|ph| ph.coherence(&target_x)).unwrap_or(-1.0);
                let cy = p.phase_for("y").map(|ph| ph.coherence(&target_y)).unwrap_or(-1.0);
                if cx >= threshold && cy >= threshold {
                    Some((p, cx * cy))
                } else {
                    None
                }
            })
            .collect();

        let (idx_results, examined) = index.query_exact_both(&target_x, &target_y, threshold);

        let mut full_ids: Vec<&str> = full.iter().map(|(p, _)| p.id.as_str()).collect();
        let mut idx_ids: Vec<&str> = idx_results.iter().map(|(p, _)| p.id.as_str()).collect();
        full_ids.sort();
        idx_ids.sort();

        let correct = full_ids == idx_ids;
        if !correct { all_correct = false; }

        let selectivity = examined as f64 / n_entities as f64 * 100.0;
        println!("    target=({:.0}°,{:.0}°): found={}, examined={}/{} ({:.1}%), correct={}",
            dx, dy, idx_results.len(), examined, n_entities, selectivity,
            if correct { "YES" } else { "NO" });
    }

    // === Part 2: Exact+Harmonic compound queries ===
    println!();
    println!("  [Exact+Harmonic Compound Queries (exact x @ 0.95, harmonic n=3 y @ 0.85)]");
    let targets_2 = [(0.0, 0.0), (120.0, 60.0), (270.0, 180.0)];
    let threshold_a = 0.95;
    let threshold_b = 0.85;
    let harmonic = 3u32;

    for &(dx, dy) in &targets_2 {
        let target_x = Phase::from_degrees(dx);
        let target_y = Phase::from_degrees(dy);

        // Ground truth: full scan
        let full: Vec<(&WavePacket, f64)> = field.packets.iter()
            .filter_map(|p| {
                let cx = p.phase_for("x").map(|ph| ph.coherence(&target_x)).unwrap_or(-1.0);
                let cy = p.phase_for("y").map(|ph| ph.harmonic_coherence(&target_y, harmonic)).unwrap_or(-1.0);
                if cx >= threshold_a && cy >= threshold_b {
                    Some((p, cx * cy))
                } else {
                    None
                }
            })
            .collect();

        let (idx_results, examined) = index.query_exact_harmonic(
            &target_x, threshold_a, &target_y, harmonic, threshold_b,
        );

        let mut full_ids: Vec<&str> = full.iter().map(|(p, _)| p.id.as_str()).collect();
        let mut idx_ids: Vec<&str> = idx_results.iter().map(|(p, _)| p.id.as_str()).collect();
        full_ids.sort();
        idx_ids.sort();

        let correct = full_ids == idx_ids;
        if !correct { all_correct = false; }

        let selectivity = examined as f64 / n_entities as f64 * 100.0;
        println!("    target=({:.0}°,{:.0}°) n={}: found={}, examined={}/{} ({:.1}%), correct={}",
            dx, dy, harmonic, idx_results.len(), examined, n_entities, selectivity,
            if correct { "YES" } else { "NO" });
    }

    // === Part 3: 2D vs 1D selectivity comparison ===
    println!();
    println!("  [2D vs 1D Selectivity Comparison]");

    let mut idx_1d = BucketIndex::new("x", bucket_count);
    for p in &field.packets {
        idx_1d.insert(p.clone());
    }

    let target_x = Phase::from_degrees(45.0);
    let target_y = Phase::from_degrees(100.0);

    let (_, examined_1d) = idx_1d.query_exact(&target_x, threshold);
    let (_, examined_2d) = index.query_exact_both(&target_x, &target_y, threshold);

    let sel_1d = examined_1d as f64 / n_entities as f64 * 100.0;
    let sel_2d = examined_2d as f64 / n_entities as f64 * 100.0;
    let improvement = if examined_2d > 0 { examined_1d as f64 / examined_2d as f64 } else { f64::INFINITY };

    println!("    1D index (x only):  examined={}/{} ({:.1}%)", examined_1d, n_entities, sel_1d);
    println!("    2D index (x and y): examined={}/{} ({:.1}%)", examined_2d, n_entities, sel_2d);
    println!("    Improvement factor: {:.1}x fewer entities examined", improvement);

    println!();
    println!("  Key insight: each indexed dimension narrows independently.");
    println!("  2D torus selectivity ~ (1D selectivity)^2 — multiplicative, not additive.");

    let pass = all_correct;
    println!();
    println!("  All compound queries match full scan: {}", all_correct);
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 20: Dynamic Mutation — Insert / Remove / Update
// =============================================================================
pub fn test_20_dynamic_mutation() -> bool {
    println!("--- Test 20: Dynamic Mutation (Insert / Remove / Update) ---");
    println!();

    // Hypothesis: a phase-indexed structure supports dynamic mutations
    // (insert, remove, update) while maintaining query correctness.
    // The circle doesn't need rebuilding — mutations are local operations.

    let n_initial = 200usize;
    let bucket_count = 60u32;
    let golden_angle = 137.50776405003785_f64;

    // Build initial index and ground truth
    let mut index = BucketIndex::new("pos", bucket_count);
    let mut truth: Vec<Option<WavePacket>> = Vec::new(); // None = removed

    for i in 0..n_initial {
        let deg = (i as f64 * golden_angle) % 360.0;
        let packet = WavePacket::new(&format!("e_{i}"))
            .with_attr("pos", Phase::from_degrees(deg));
        index.insert(packet.clone());
        truth.push(Some(packet));
    }

    println!("  Initial entities: {n_initial}, Buckets: {bucket_count}");
    println!("  Active: {}", index.active_count());
    println!();

    let threshold = 0.95;
    let mut all_correct = true;

    // Helper closure: build ground truth results for a query
    fn truth_query_exact<'a>(truth: &'a [Option<WavePacket>], target: &Phase, threshold: f64) -> Vec<&'a str> {
        let mut ids: Vec<&str> = truth.iter()
            .filter_map(|opt| opt.as_ref())
            .filter(|p| p.phase_for("pos").map(|ph| ph.coherence(target) >= threshold).unwrap_or(false))
            .map(|p| p.id.as_str())
            .collect();
        ids.sort();
        ids
    }

    fn truth_query_harmonic<'a>(truth: &'a [Option<WavePacket>], target: &Phase, n: u32, threshold: f64) -> Vec<&'a str> {
        let mut ids: Vec<&str> = truth.iter()
            .filter_map(|opt| opt.as_ref())
            .filter(|p| p.phase_for("pos").map(|ph| ph.harmonic_coherence(target, n) >= threshold).unwrap_or(false))
            .map(|p| p.id.as_str())
            .collect();
        ids.sort();
        ids
    }

    // === Phase 1: Remove 50 entities ===
    println!("  [Phase 1: Remove 50 entities]");
    let remove_ids: Vec<String> = (0..50).map(|i| format!("e_{}", i * 4)).collect();

    for id in &remove_ids {
        let removed = index.remove_by_id(id);
        if !removed { all_correct = false; }
        // Mirror in ground truth
        if let Some(pos) = truth.iter().position(|opt| {
            opt.as_ref().map(|p| p.id.as_str()) == Some(id.as_str())
        }) {
            truth[pos] = None;
        }
    }

    let active_after_remove = index.active_count();
    println!("    Removed: {}, Active: {}", remove_ids.len(), active_after_remove);

    // Verify: double-remove returns false
    let double_remove = index.remove_by_id("e_0");
    let double_ok = !double_remove;
    if !double_ok { all_correct = false; }
    println!("    Double-remove returns false: {}", if double_ok { "YES" } else { "NO" });

    // Verify queries after removal
    let target = Phase::from_degrees(45.0);
    let (idx_results, _) = index.query_exact(&target, threshold);
    let mut idx_ids: Vec<&str> = idx_results.iter().map(|(p, _)| p.id.as_str()).collect();
    idx_ids.sort();
    let truth_ids = truth_query_exact(&truth, &target, threshold);

    let remove_correct = idx_ids == truth_ids;
    if !remove_correct { all_correct = false; }
    println!("    Query after remove: found={}, correct={}", idx_results.len(),
        if remove_correct { "YES" } else { "NO" });

    // === Phase 2: Insert 30 new entities ===
    println!();
    println!("  [Phase 2: Insert 30 new entities]");

    for i in 0..30 {
        let deg = (i as f64 * 12.345 + 77.0) % 360.0;
        let packet = WavePacket::new(&format!("new_{i}"))
            .with_attr("pos", Phase::from_degrees(deg));
        index.insert(packet.clone());
        truth.push(Some(packet));
    }

    let active_after_insert = index.active_count();
    println!("    Inserted: 30, Active: {}", active_after_insert);

    // Verify queries after insertion
    let (idx_results, _) = index.query_exact(&target, threshold);
    let mut idx_ids: Vec<&str> = idx_results.iter().map(|(p, _)| p.id.as_str()).collect();
    idx_ids.sort();
    let truth_ids = truth_query_exact(&truth, &target, threshold);

    let insert_correct = idx_ids == truth_ids;
    if !insert_correct { all_correct = false; }
    println!("    Query after insert: found={}, correct={}", idx_results.len(),
        if insert_correct { "YES" } else { "NO" });

    // === Phase 3: Update 20 entities (reposition) ===
    println!();
    println!("  [Phase 3: Update 20 entities (reposition)]");

    // Pick IDs guaranteed to still exist (odd numbers, not removed in Phase 1)
    let update_ids: Vec<String> = (0..20).map(|i| format!("e_{}", i * 2 + 1)).collect();
    let mut updates_applied = 0;

    for (i, id) in update_ids.iter().enumerate() {
        let new_deg = (i as f64 * 18.0 + 5.0) % 360.0;
        let new_packet = WavePacket::new(id)
            .with_attr("pos", Phase::from_degrees(new_deg));

        let found = index.update(id, new_packet.clone());
        if found { updates_applied += 1; }

        // Mirror in ground truth: remove old, add new
        if let Some(pos) = truth.iter().position(|opt| {
            opt.as_ref().map(|p| p.id.as_str()) == Some(id.as_str())
        }) {
            truth[pos] = None;
        }
        truth.push(Some(new_packet));
    }

    let active_after_update = index.active_count();
    println!("    Update attempts: {}, Applied: {}, Active: {}", update_ids.len(), updates_applied, active_after_update);

    // Verify queries at multiple targets after update
    let test_targets = [0.0, 45.0, 90.0, 180.0, 270.0];
    let mut multi_correct = 0;

    for &deg in &test_targets {
        let t = Phase::from_degrees(deg);
        let (idx_results, _) = index.query_exact(&t, threshold);
        let mut idx_ids: Vec<&str> = idx_results.iter().map(|(p, _)| p.id.as_str()).collect();
        idx_ids.sort();
        let truth_ids = truth_query_exact(&truth, &t, threshold);
        if idx_ids == truth_ids { multi_correct += 1; } else { all_correct = false; }
    }

    println!("    Post-update exact queries: {}/{} correct", multi_correct, test_targets.len());

    // === Phase 4: Harmonic queries after all mutations ===
    println!();
    println!("  [Phase 4: Harmonic queries after all mutations]");

    let harm_targets = [0.0, 120.0, 240.0];
    let harm_threshold = 0.85;
    let mut harm_correct = 0;

    for &deg in &harm_targets {
        let t = Phase::from_degrees(deg);
        let (idx_results, _) = index.query_harmonic(&t, 3, harm_threshold);
        let mut idx_ids: Vec<&str> = idx_results.iter().map(|(p, _)| p.id.as_str()).collect();
        idx_ids.sort();
        let truth_ids = truth_query_harmonic(&truth, &t, 3, harm_threshold);
        if idx_ids == truth_ids { harm_correct += 1; } else { all_correct = false; }
    }

    println!("    Harmonic n=3 post-mutation: {}/{} correct", harm_correct, harm_targets.len());

    // === Summary ===
    println!();
    println!("  Mutation summary:");
    println!("    {} initial - {} removed + {} inserted - {} repositioned = {} active",
        n_initial, remove_ids.len(), 30, updates_applied, active_after_update);
    println!("  Key insight: mutations are local operations on the circle.");
    println!("  Remove = tombstone + bucket cleanup. Update = remove + re-insert.");
    println!("  No global rebuild needed. Queries remain correct throughout.");

    let pass = all_correct;
    println!();
    println!("  All queries correct after mutations: {}", all_correct);
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}
