use crate::wave::{Phase, WavePacket};
use crate::field::ResonanceField;

// =============================================================================
// Test 14: Harmonic Orthogonality (No Cross-Talk Between Harmonics)
// =============================================================================
pub fn test_14_harmonic_orthogonality() -> bool {
    println!("--- Test 14: Harmonic Orthogonality ---");

    let mut field = ResonanceField::new();
    let test_angles = [0.0, 60.0, 72.0, 90.0, 120.0, 180.0, 240.0, 270.0, 288.0, 300.0];
    for &deg in &test_angles {
        field.add(WavePacket::new(&format!("pos_{deg}"))
            .with_attr("angle", Phase::from_degrees(deg)));
    }

    let target = Phase::from_degrees(0.0);
    let threshold = 0.95;

    // n=3 should find 0°, 120°, 240° and NOTHING ELSE
    let h3 = field.query_harmonic("angle", &target, 3, threshold);
    let h3_ids: Vec<&str> = h3.iter().map(|(p, _)| p.id.as_str()).collect();

    // n=4 should find 0°, 90°, 180°, 270° and NOTHING ELSE
    let h4 = field.query_harmonic("angle", &target, 4, threshold);
    let h4_ids: Vec<&str> = h4.iter().map(|(p, _)| p.id.as_str()).collect();

    // n=6 should find 0°, 60°, 120°, 180°, 240°, 300° and NOTHING ELSE
    let h6 = field.query_harmonic("angle", &target, 6, threshold);
    let h6_ids: Vec<&str> = h6.iter().map(|(p, _)| p.id.as_str()).collect();

    // n=5 should find 0°, 72°, 144°, 216°, 288° — we have 0°, 72°, 288°
    let h5 = field.query_harmonic("angle", &target, 5, threshold);
    let h5_ids: Vec<&str> = h5.iter().map(|(p, _)| p.id.as_str()).collect();

    println!("  n=3 (120°): {:?}", h3_ids);
    println!("  n=4 (90°):  {:?}", h4_ids);
    println!("  n=5 (72°):  {:?}", h5_ids);
    println!("  n=6 (60°):  {:?}", h6_ids);

    // Cross-talk checks
    let h3_no_90 = !h3_ids.contains(&"pos_90") && !h3_ids.contains(&"pos_270");
    let h3_no_60 = !h3_ids.contains(&"pos_60") && !h3_ids.contains(&"pos_300");
    let h4_no_120 = !h4_ids.contains(&"pos_120") && !h4_ids.contains(&"pos_240");
    let h4_no_60 = !h4_ids.contains(&"pos_60") && !h4_ids.contains(&"pos_300");

    // Inclusion checks
    let h3_has_120 = h3_ids.contains(&"pos_120") && h3_ids.contains(&"pos_240");
    let h4_has_90 = h4_ids.contains(&"pos_90") && h4_ids.contains(&"pos_270");
    let h5_has_72 = h5_ids.contains(&"pos_72") && h5_ids.contains(&"pos_288");

    println!("\n  Cross-talk checks:");
    println!("    n=3 excludes 90°/270°:  {}", h3_no_90);
    println!("    n=3 excludes 60°/300°:  {}", h3_no_60);
    println!("    n=4 excludes 120°/240°: {}", h4_no_120);
    println!("    n=4 excludes 60°/300°:  {}", h4_no_60);
    println!("  Inclusion checks:");
    println!("    n=3 finds 120°/240°: {}", h3_has_120);
    println!("    n=4 finds 90°/270°:  {}", h4_has_90);
    println!("    n=5 finds 72°/288°:  {}", h5_has_72);

    let pass = h3_no_90 && h3_no_60 && h4_no_120 && h4_no_60
        && h3_has_120 && h4_has_90 && h5_has_72;

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 15: Phase Wraparound at 0°/360° Boundary
// =============================================================================
pub fn test_15_wraparound() -> bool {
    println!("--- Test 15: Phase Wraparound at 0°/360° Boundary ---");

    let a = Phase::from_degrees(1.0);
    let b = Phase::from_degrees(359.0);
    let dist = a.distance_degrees(&b);

    println!("  1° to 359°: distance = {:.4}° (expected: 2.0°)", dist);
    let dist_ok = (dist - 2.0).abs() < 0.001;

    let coh = a.coherence(&b);
    println!("  1° to 359°: coherence = {:.6} (expected: ~0.9994)", coh);
    let coh_ok = coh > 0.999;

    let target = Phase::from_degrees(0.0);
    let score_1 = target.fuzzy_match(&a, 0.0, 8.0);
    let score_359 = target.fuzzy_match(&b, 0.0, 8.0);
    println!("  Fuzzy(0° → 1°, target=0°, orb=8°):   {:.6}", score_1);
    println!("  Fuzzy(0° → 359°, target=0°, orb=8°):  {:.6}", score_359);
    let fuzzy_ok = score_1 > 0.97 && score_359 > 0.97
        && (score_1 - score_359).abs() < 0.0001;

    let dir_1_to_359 = a.directed_distance_degrees(&b);
    let dir_359_to_1 = b.directed_distance_degrees(&a);
    println!("  Directed 1° → 359°: {:.4}° (expected: 358.0°)", dir_1_to_359);
    println!("  Directed 359° → 1°: {:.4}° (expected: 2.0°)", dir_359_to_1);
    let dir_ok = (dir_1_to_359 - 358.0).abs() < 0.001 && (dir_359_to_1 - 2.0).abs() < 0.001;

    let mut field = ResonanceField::new();
    for &deg in &[357.0, 358.0, 359.0, 0.0, 1.0, 2.0, 3.0, 180.0] {
        field.add(WavePacket::new(&format!("pos_{deg}"))
            .with_attr("angle", Phase::from_degrees(deg)));
    }
    let near_zero = field.query_fuzzy("angle", &target, 0.0, 5.0);
    let near_ids: Vec<&str> = near_zero.iter().map(|(p, _)| p.id.as_str()).collect();
    println!("  Entities within 5° of 0°: {:?}", near_ids);
    let field_ok = near_ids.contains(&"pos_0") && near_ids.contains(&"pos_1")
        && near_ids.contains(&"pos_2") && near_ids.contains(&"pos_3")
        && near_ids.contains(&"pos_357") && near_ids.contains(&"pos_358")
        && near_ids.contains(&"pos_359") && !near_ids.contains(&"pos_180");

    let pass = dist_ok && coh_ok && fuzzy_ok && dir_ok && field_ok;
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 16: Scale Resolution — 360 Distinct Values, Zero False Positives
// =============================================================================
pub fn test_16_scale_resolution() -> bool {
    println!("--- Test 16: Scale Resolution (360 Values) ---");

    let bucket_count = 360;
    let mut field = ResonanceField::new();

    for i in 0..360u64 {
        field.add(WavePacket::new(&format!("val_{i}"))
            .with_attr("v", Phase::from_value(i, bucket_count)));
    }

    let bucket_angle_cos = (2.0 * std::f64::consts::PI / bucket_count as f64).cos();
    let threshold = (1.0 + bucket_angle_cos) / 2.0;
    println!("  cos(1°) = {:.6}, threshold = {:.6}", bucket_angle_cos, threshold);

    let mut perfect = true;
    let mut total_matches = 0;
    let mut false_positives = 0;

    for i in 0..360u64 {
        let target = Phase::from_value(i, bucket_count);
        let results = field.query_exact("v", &target, threshold);
        total_matches += results.len();
        if results.len() != 1 {
            if i < 5 || i > 355 {
                println!("  val_{}: got {} matches (expected 1)", i, results.len());
            }
            perfect = false;
            false_positives += results.len().saturating_sub(1);
        } else if results[0].0.id != format!("val_{i}") {
            println!("  val_{}: wrong match! got {}", i, results[0].0.id);
            perfect = false;
            false_positives += 1;
        }
    }

    println!("  360 queries, {} total matches, {} false positives", total_matches, false_positives);
    println!("  Perfect resolution (each query → exactly 1 correct match): {}", perfect);

    let target_0 = Phase::from_value(0, bucket_count);
    let h3_results = field.query_harmonic("v", &target_0, 3, 0.99);
    let h3_ids: Vec<&str> = h3_results.iter().map(|(p, _)| p.id.as_str()).collect();
    let h3_has_centers = h3_ids.contains(&"val_0") && h3_ids.contains(&"val_120") && h3_ids.contains(&"val_240");
    println!("  Harmonic n=3 from val_0: {} matches (5 per group × 3 groups), centers present: {}", h3_results.len(), h3_has_centers);

    let h4_results = field.query_harmonic("v", &target_0, 4, 0.99);
    let h4_ids: Vec<&str> = h4_results.iter().map(|(p, _)| p.id.as_str()).collect();
    let h4_has_centers = h4_ids.contains(&"val_0") && h4_ids.contains(&"val_90")
        && h4_ids.contains(&"val_180") && h4_ids.contains(&"val_270");
    println!("  Harmonic n=4 from val_0: {} matches (5 per group × 4 groups), centers present: {}", h4_results.len(), h4_has_centers);

    let h3_nyquist = (3.0 * 2.0 * std::f64::consts::PI / bucket_count as f64).cos();
    let h4_nyquist = (4.0 * 2.0 * std::f64::consts::PI / bucket_count as f64).cos();
    println!("  Nyquist floor at n=3: cos(3°) = {:.6} (threshold must exceed this for single-value precision)", h3_nyquist);
    println!("  Nyquist floor at n=4: cos(4°) = {:.6}", h4_nyquist);

    let pass = perfect && h3_has_centers && h3_results.len() == 15
        && h4_has_centers && h4_results.len() == 20;
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}
