use crate::wave::{Phase, WavePacket};
use crate::field::{ResonanceField, BucketIndex};

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
