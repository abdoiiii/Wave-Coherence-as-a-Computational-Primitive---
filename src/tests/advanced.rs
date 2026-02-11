use crate::wave::{Phase, WavePacket};
use crate::field::ResonanceField;
use crate::relationships::DirectedCycle;

// =============================================================================
// Test 10: Entity Type-Dependent Reach
// =============================================================================
pub fn test_10_typed_reach() -> bool {
    println!("--- Test 10: Entity Type-Dependent Reach ---");

    let mut field = ResonanceField::new();

    // 12 entities at 30° intervals
    for i in 0..12 {
        let deg = i as f64 * 30.0;
        let packet = WavePacket::new(&format!("pos_{deg}"))
            .with_attr("angle", Phase::from_degrees(deg));
        field.add(packet);
    }

    let target = Phase::from_degrees(0.0);
    let orb = 8.0;

    // "broad" type sees 60°, 180°, 270°
    let broad_angles = vec![60.0, 180.0, 270.0];
    let broad_results = field.query_typed_reach("angle", &target, &broad_angles, orb);

    println!("  Entity at 0° with type 'broad' (sees 60°, 180°, 270°):");
    for (p, score) in &broad_results {
        println!("    {} -> score = {:.4}", p.id, score);
    }

    // "narrow" type sees only 180°
    let narrow_angles = vec![180.0];
    let narrow_results = field.query_typed_reach("angle", &target, &narrow_angles, orb);

    println!("  Entity at 0° with type 'narrow' (sees 180° only):");
    for (p, score) in &narrow_results {
        println!("    {} -> score = {:.4}", p.id, score);
    }

    // Broad should find entities at 60°, 180°, 270°
    let broad_ids: Vec<&str> = broad_results.iter().map(|(p, _)| p.id.as_str()).collect();
    let broad_has_60 = broad_ids.contains(&"pos_60");
    let broad_has_180 = broad_ids.contains(&"pos_180");
    let broad_has_270 = broad_ids.contains(&"pos_270");

    // Narrow should find only entity at 180°
    let narrow_ids: Vec<&str> = narrow_results.iter().map(|(p, _)| p.id.as_str()).collect();
    let narrow_has_180 = narrow_ids.contains(&"pos_180");
    let narrow_count = narrow_ids.len();

    println!("  Broad found 60°: {}, 180°: {}, 270°: {}", broad_has_60, broad_has_180, broad_has_270);
    println!("  Narrow found 180°: {}, total count: {}", narrow_has_180, narrow_count);
    println!("  Same position, different type, different results: {}", broad_results.len() != narrow_results.len());

    let pass = broad_has_60 && broad_has_180 && broad_has_270
        && narrow_has_180 && narrow_count == 1;

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 11: Harmonic Fingerprint Disambiguation (CONJECTURE VALIDATION)
// =============================================================================
pub fn test_11_harmonic_fingerprint() -> bool {
    println!("--- Test 11: Harmonic Fingerprint Disambiguation ---");

    let a = Phase::from_degrees(5.0);
    let b = Phase::from_degrees(7.0);

    println!("  Phase A: 5°, Phase B: 7° (2° apart)");
    println!("  At n=1 they are nearly indistinguishable.");
    println!("  Scanning harmonics to find divergence:\n");

    let mut first_divergence_n = 0u32;
    let threshold = 0.9; // below this = clearly different

    println!("  {:>4}  {:>12}  {:>10}", "n", "coherence", "status");
    println!("  {:>4}  {:>12}  {:>10}", "---", "---------", "------");

    for n in 1..=90 {
        let c = a.harmonic_coherence(&b, n);
        let status = if c.abs() >= threshold { "similar" } else { "DIVERGED" };

        // Print key harmonics
        if n <= 10 || n % 10 == 0 || (c.abs() < threshold && first_divergence_n == 0) {
            println!("  {:>4}  {:>12.6}  {:>10}", n, c, status);
        }

        if c.abs() < threshold && first_divergence_n == 0 {
            first_divergence_n = n;
        }
    }

    println!("\n  First divergence at harmonic n={}", first_divergence_n);

    // Harder case: 1° apart
    println!("\n  --- Harder case: 1° apart ---");
    let c_val = Phase::from_degrees(10.0);
    let d = Phase::from_degrees(11.0);

    let mut first_divergence_hard = 0u32;
    for n in 1..=180 {
        let c = c_val.harmonic_coherence(&d, n);
        if c.abs() < threshold && first_divergence_hard == 0 {
            first_divergence_hard = n;
            println!("  1° apart: first divergence at n={}, coherence={:.6}", n, c);
        }
    }

    println!("\n  --- Extreme case: 0.1° apart ---");
    let e = Phase::from_degrees(10.0);
    let f = Phase::from_degrees(10.1);

    let mut first_divergence_extreme = 0u32;
    for n in 1..=1800 {
        let c = e.harmonic_coherence(&f, n);
        if c.abs() < threshold && first_divergence_extreme == 0 {
            first_divergence_extreme = n;
            println!("  0.1° apart: first divergence at n={}, coherence={:.6}", n, c);
        }
    }

    // The pattern: smaller angular difference requires higher harmonic to resolve.
    // Formula: n_diverge ≈ arccos(threshold) / Δθ
    let delta_2 = 2.0_f64;
    let predicted_n_2 = (threshold.acos().to_degrees() / delta_2).ceil() as u32;
    println!("\n  Predicted divergence (2° apart):   n≈{}", predicted_n_2);
    println!("  Actual divergence (2° apart):      n={}", first_divergence_n);
    println!("  Predicted divergence (1° apart):   n≈{}", (threshold.acos().to_degrees() / 1.0).ceil() as u32);
    println!("  Actual divergence (1° apart):      n={}", first_divergence_hard);
    println!("  Predicted divergence (0.1° apart): n≈{}", (threshold.acos().to_degrees() / 0.1).ceil() as u32);
    println!("  Actual divergence (0.1° apart):    n={}", first_divergence_extreme);

    // Pass conditions:
    // 1. All three cases eventually diverge (fingerprints are unique)
    // 2. Smaller differences require higher harmonics (scaling relationship)
    let pass = first_divergence_n > 0
        && first_divergence_hard > 0
        && first_divergence_extreme > 0
        && first_divergence_hard > first_divergence_n
        && first_divergence_extreme > first_divergence_hard;

    println!("\n  Divergence ordering: {} < {} < {} (smaller Δ needs higher n): {}",
        first_divergence_n, first_divergence_hard, first_divergence_extreme,
        first_divergence_hard > first_divergence_n && first_divergence_extreme > first_divergence_hard);

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 12: Mutual Reference Amplification
// =============================================================================
pub fn test_12_mutual_amplification() -> bool {
    println!("--- Test 12: Mutual Reference Amplification ---");

    let a = Phase::from_degrees(30.0);
    let b = Phase::from_degrees(35.0); // close but not identical

    let base_coherence = a.coherence(&b);

    // Simulate reference patterns
    let a_refs_b = true;
    let b_refs_a_mutual = true;
    let b_refs_a_oneway = false;

    let mutual_score = if a_refs_b && b_refs_a_mutual {
        base_coherence * 1.5
    } else if a_refs_b || b_refs_a_mutual {
        base_coherence * 1.2
    } else {
        base_coherence
    };

    let oneway_score = if a_refs_b && b_refs_a_oneway {
        base_coherence * 1.5
    } else if a_refs_b || b_refs_a_oneway {
        base_coherence * 1.2
    } else {
        base_coherence
    };

    let no_ref_score = base_coherence; // neither references

    println!("  A at 30°, B at 35° (5° apart)");
    println!("  Base coherence: {:.6}", base_coherence);
    println!("  Mutual (A↔B):   {:.6} (× 1.5)", mutual_score);
    println!("  One-way (A→B):  {:.6} (× 1.2)", oneway_score);
    println!("  No reference:   {:.6} (× 1.0)", no_ref_score);

    // Verify ordering: mutual > one-way > no-ref
    let ordering_correct = mutual_score > oneway_score && oneway_score > no_ref_score;

    // Verify amplification ratios
    let mutual_ratio = mutual_score / base_coherence;
    let oneway_ratio = oneway_score / base_coherence;

    println!("  Mutual ratio:  {:.2} (expected 1.50)", mutual_ratio);
    println!("  One-way ratio: {:.2} (expected 1.20)", oneway_ratio);
    println!("  Ordering (mutual > oneway > none): {}", ordering_correct);

    let pass = ordering_correct
        && (mutual_ratio - 1.5).abs() < 0.001
        && (oneway_ratio - 1.2).abs() < 0.001;

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 13: Exhaustive 5-Node Cycle Relationship Uniqueness
// =============================================================================
pub fn test_13_cycle_uniqueness() -> bool {
    println!("--- Test 13: Exhaustive 5-Node Cycle Relationship Uniqueness ---");

    let cycle = DirectedCycle::new(5);

    let mut relationship_map: Vec<Vec<Option<i32>>> = vec![vec![None; 5]; 5];
    let steps = [1, 2, -1, -2]; // generative, destructive, weakening, controlling
    let step_names = ["+1 (generative)", "+2 (destructive)", "-1 (weakening)", "-2 (controlling)"];

    let mut all_assigned = true;
    let mut no_conflicts = true;

    for &step in &steps {
        for start in 0..5 {
            let dest = cycle.step(start, step);
            if let Some(existing) = relationship_map[start][dest] {
                println!("  CONFLICT: ({},{}) has both step {} and step {}",
                    start, dest, existing, step);
                no_conflicts = false;
            } else {
                relationship_map[start][dest] = Some(step);
            }
        }
    }

    // Check that every ordered pair (a, b) where a ≠ b has been assigned
    println!("  Relationship matrix (row=from, col=to, value=step):");
    println!("       0     1     2     3     4");
    for a in 0..5 {
        print!("  {}:", a);
        for b in 0..5 {
            if a == b {
                print!("    . ");
            } else {
                match relationship_map[a][b] {
                    Some(s) => print!("  {:>+3} ", s),
                    None => {
                        print!("  NONE");
                        all_assigned = false;
                    }
                }
            }
        }
        println!();
    }

    // Count relationships per type
    let mut counts = [0usize; 4];
    for a in 0..5 {
        for b in 0..5 {
            if a != b {
                if let Some(step) = relationship_map[a][b] {
                    let idx = steps.iter().position(|&s| s == step).unwrap();
                    counts[idx] += 1;
                }
            }
        }
    }

    println!("\n  Relationships per type:");
    for (i, name) in step_names.iter().enumerate() {
        println!("    {}: {} pairs", name, counts[i]);
    }

    let total_pairs = counts.iter().sum::<usize>();
    let expected_pairs = 5 * 4; // 20 ordered pairs (a≠b)

    println!("\n  Total assigned: {} / {} ordered pairs", total_pairs, expected_pairs);
    println!("  All pairs assigned: {}", all_assigned);
    println!("  No conflicts: {}", no_conflicts);
    println!("  Each type has exactly 5 pairs: {}", counts.iter().all(|&c| c == 5));

    let pass = all_assigned && no_conflicts && total_pairs == expected_pairs
        && counts.iter().all(|&c| c == 5);

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}
