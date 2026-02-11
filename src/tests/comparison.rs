use crate::wave::{Phase, WavePacket};
use crate::field::ResonanceField;

// =============================================================================
// Test 8: Wave vs Linear Scan (Correctness)
// =============================================================================
pub fn test_8_wave_vs_linear() -> bool {
    println!("--- Test 8: Wave vs Linear Scan (Correctness) ---");

    let bucket_count = 100;
    let entity_count = 1000;
    let target_value: u64 = 42;

    // Generate entities with deterministic "random" values
    let mut field = ResonanceField::new();
    let mut values: Vec<(String, u64)> = vec![];

    for i in 0..entity_count {
        // Simple deterministic hash to spread values
        let val = (i * 37 + 13) % bucket_count as u64;
        let id = format!("entity_{i}");
        values.push((id.clone(), val));
        let packet = WavePacket::new(&id)
            .with_attr("value", Phase::from_value(val, bucket_count));
        field.add(packet);
    }

    // Method A: Linear scan with value comparison
    let linear_results: Vec<String> = values.iter()
        .filter(|(_, v)| *v == target_value)
        .map(|(id, _)| id.clone())
        .collect();

    // Method B: Coherence scan
    // With 100 buckets, each is 3.6°. cos(3.6°)=0.998, so threshold must be > 0.998
    // to avoid catching adjacent buckets. Use 0.9999 for single-bucket precision.
    let target_phase = Phase::from_value(target_value, bucket_count);
    let wave_results: Vec<String> = field.query_exact("value", &target_phase, 0.9999)
        .iter()
        .map(|(p, _)| p.id.clone())
        .collect();

    println!("  {} entities, bucket_count={}, target_value={}", entity_count, bucket_count, target_value);
    println!("  Linear scan found: {} matches", linear_results.len());
    println!("  Wave scan found:   {} matches", wave_results.len());

    // Check both sets are identical
    let mut linear_sorted = linear_results.clone();
    let mut wave_sorted = wave_results.clone();
    linear_sorted.sort();
    wave_sorted.sort();

    let pass = linear_sorted == wave_sorted;

    if !pass {
        println!("  MISMATCH!");
        println!("  Linear: {:?}", &linear_sorted[..linear_sorted.len().min(10)]);
        println!("  Wave:   {:?}", &wave_sorted[..wave_sorted.len().min(10)]);
    } else {
        println!("  Result sets are IDENTICAL");
    }

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 9: Harmonic Query vs JOIN (Value Proposition)
// =============================================================================
pub fn test_9_harmonic_vs_join() -> bool {
    println!("--- Test 9: Harmonic Query vs JOIN ---");

    let mut field = ResonanceField::new();

    // 4 groups of 25 entities, at 0°, 90°, 120°, 240° (with slight scatter)
    let group_centers = [0.0, 90.0, 120.0, 240.0];
    let group_names = ["group_0", "group_90", "group_120", "group_240"];

    for (gi, &center) in group_centers.iter().enumerate() {
        for j in 0..25 {
            // Small scatter within ±2.4° of center
            let offset = (j as f64 - 12.0) * 0.2;
            let deg = center + offset;
            let id = format!("{}_{}", group_names[gi], j);
            let packet = WavePacket::new(&id)
                .with_attr("pos", Phase::from_degrees(deg));
            field.add(packet);
        }
    }

    // Target entity: at 3° (in the 0° group)
    let target = Phase::from_degrees(3.0);

    // Harmonic scan: 3rd harmonic finds both 120° and 240° automatically
    let harmonic_results = field.query_harmonic("pos", &target, 3, 0.85);

    // Categorize results by group
    let mut group_counts = [0usize; 4];
    for (p, _) in &harmonic_results {
        for (gi, name) in group_names.iter().enumerate() {
            if p.id.starts_with(name) {
                group_counts[gi] += 1;
            }
        }
    }

    println!("  Target: entity at 3° (in group_0)");
    println!("  3rd harmonic scan results by group:");
    for (gi, name) in group_names.iter().enumerate() {
        println!("    {}: {} entities found", name, group_counts[gi]);
    }
    println!("  Total harmonic matches: {}", harmonic_results.len());

    let found_120 = group_counts[2] > 0;
    let found_240 = group_counts[3] > 0;
    let not_found_90 = group_counts[1] == 0;

    println!("  Found group_120: {}", found_120);
    println!("  Found group_240: {}", found_240);
    println!("  Excluded group_90: {}", not_found_90);
    println!("  Key insight: Single harmonic scan found BOTH 120° and 240° groups");
    println!("  Traditional approach would need 2 JOINs for the same result");

    let pass = found_120 && found_240 && not_found_90;
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}
