use crate::wave::{Phase, WavePacket};
use crate::field::ResonanceField;

// =============================================================================
// Test 17: Density Scaling and Capacity Limits
// =============================================================================
pub fn test_17_scaling_limits() -> bool {
    println!("--- Test 17: Density Scaling and Capacity Limits ---");
    println!();

    // Test how the system behaves as the ratio of objects to buckets increases.
    // For each (N, B) configuration: place N objects on a B-bucket circle,
    // measure exact-match precision, minimum pairwise separation, the harmonic
    // needed to resolve the closest pair, and whether triadic (n=3) detection
    // remains noise-free.
    //
    // Object placement uses the golden angle (≈137.508°) for irrational spacing,
    // avoiding artificial grid alignment.

    use std::f64::consts::PI;

    struct ScenarioResult {
        name: &'static str,
        n_objects: usize,
        buckets: u32,
        density_pct: f64,       // N/B as percentage
        min_separation_deg: f64,
        max_harmonic_needed: u32,
        exact_match_ok: bool,   // can every object be uniquely identified?
        triadic_clean: bool,    // does n=3 harmonic find correct groups without noise?
    }

    let scenarios: Vec<(&str, usize, u32)> = vec![
        ("7 in 12", 7, 12),
        ("9 in 27", 9, 27),
        ("12 in 12 (saturated)", 12, 12),
        ("20 in 60", 20, 60),
        ("50 in 360", 50, 360),
        ("100 in 360", 100, 360),
        ("200 in 360", 200, 360),
        ("360 in 360 (saturated)", 360, 360),
    ];

    let mut results: Vec<ScenarioResult> = Vec::new();

    for (name, n_objects, buckets) in &scenarios {
        let n = *n_objects;
        let b = *buckets;

        let golden_angle = 137.50776405003785_f64;
        let mut positions_deg: Vec<f64> = Vec::new();
        for i in 0..n {
            let deg = (i as f64 * golden_angle) % 360.0;
            positions_deg.push(deg);
        }

        let mut field = ResonanceField::new();
        for (i, &deg) in positions_deg.iter().enumerate() {
            field.add(WavePacket::new(&format!("obj_{i}"))
                .with_attr("pos", Phase::from_degrees(deg)));
        }

        // 1. Find minimum angular separation between any pair
        let mut min_sep = 360.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let pi = Phase::from_degrees(positions_deg[i]);
                let pj = Phase::from_degrees(positions_deg[j]);
                let sep = pi.distance_degrees(&pj);
                if sep < min_sep {
                    min_sep = sep;
                }
            }
        }

        // 2. Compute maximum harmonic needed to resolve closest pair
        let threshold = 0.9_f64;
        let max_harmonic = if min_sep > 0.001 {
            (threshold.acos().to_degrees() / min_sep).ceil() as u32
        } else {
            9999
        };

        // 3. Test exact match: can each object be uniquely identified?
        let bucket_angle_cos = (2.0 * PI / b as f64).cos();
        let exact_threshold = (1.0 + bucket_angle_cos) / 2.0;

        let mut exact_ok = true;
        for (i, &deg) in positions_deg.iter().enumerate() {
            let target = Phase::from_degrees(deg);
            let matches = field.query_exact("pos", &target, exact_threshold);
            let found_self = matches.iter().any(|(p, _)| p.id == format!("obj_{i}"));
            if !found_self || matches.len() != 1 {
                exact_ok = false;
            }
        }

        // 4. Test triadic detection: query n=3 harmonic from object 0.
        let target_phase = Phase::from_degrees(positions_deg[0]);
        let triadic_results = field.query_harmonic("pos", &target_phase, 3, 0.85);

        let mut noise_triadic = 0usize;
        for (p, _) in &triadic_results {
            let p_phase = p.phase_for("pos").unwrap();
            let dist = target_phase.distance_degrees(p_phase);
            let near_0 = dist < 10.0;
            let near_120 = (dist - 120.0).abs() < 10.0;
            let near_240 = (dist - 240.0).abs() < 10.0;
            if !(near_0 || near_120 || near_240) {
                noise_triadic += 1;
            }
        }
        let triadic_clean = noise_triadic == 0;

        let density = n as f64 / b as f64 * 100.0;

        results.push(ScenarioResult {
            name,
            n_objects: n,
            buckets: b,
            density_pct: density,
            min_separation_deg: min_sep,
            max_harmonic_needed: max_harmonic,
            exact_match_ok: exact_ok,
            triadic_clean,
        });
    }

    // Print results table
    println!("  {:<30} {:>4} {:>5} {:>7} {:>9} {:>8} {:>7} {:>7}",
        "Configuration", "N", "B", "N/B %", "Min Sep", "Max n", "Exact", "Triad");
    println!("  {:<30} {:>4} {:>5} {:>7} {:>9} {:>8} {:>7} {:>7}",
        "-------------", "---", "---", "-----", "-------", "-----", "-----", "-----");

    for r in &results {
        println!("  {:<30} {:>4} {:>5} {:>6.1}% {:>8.3}° {:>8} {:>7} {:>7}",
            r.name, r.n_objects, r.buckets, r.density_pct,
            r.min_separation_deg, r.max_harmonic_needed,
            if r.exact_match_ok { "OK" } else { "FAIL" },
            if r.triadic_clean { "clean" } else { "noisy" });
    }

    // Analysis: find degradation points
    println!();
    let first_exact_fail = results.iter().find(|r| !r.exact_match_ok);
    let first_triadic_noise = results.iter().find(|r| !r.triadic_clean);

    if let Some(r) = first_exact_fail {
        println!("  Exact match degrades at: {} ({:.1}% density)", r.name, r.density_pct);
    } else {
        println!("  Exact match: no degradation across all scenarios");
    }

    if let Some(r) = first_triadic_noise {
        println!("  Triadic (n=3) noise begins at: {} ({:.1}% density)", r.name, r.density_pct);
    } else {
        println!("  Triadic detection: clean across all scenarios");
    }

    // Collision probability (birthday problem) for each configuration
    println!();
    println!("  Bucket collision probability (birthday problem approximation):");
    for r in &results {
        let n = r.n_objects as f64;
        let b = r.buckets as f64;
        let p_collision = 1.0 - (-n * (n - 1.0) / (2.0 * b)).exp();
        println!("    {}: P(collision) = {:.1}%", r.name, p_collision * 100.0);
    }

    // Design rules derived from the data
    println!();
    println!("  Design rules:");
    println!("    - Exact match requires density < 100%% (no two objects in same bucket)");
    if let Some(r) = first_triadic_noise {
        println!("    - Clean triadic detection requires density < ~{:.0}%% at threshold 0.85",
            r.density_pct);
    }
    println!("    - Resolution harmonic scales inversely with minimum separation");
    println!("    - Collision probability follows birthday problem: P ≈ 1 - e^(-N²/2B)");

    // Pass conditions
    let smallest_ok = results[0].exact_match_ok && results[0].triadic_clean;
    let degrades = results.iter().any(|r| !r.exact_match_ok || !r.triadic_clean);
    let harmonic_scales = results.first().unwrap().max_harmonic_needed
        < results.last().unwrap().max_harmonic_needed;
    let exact_only_at_saturation = results.iter()
        .all(|r| r.exact_match_ok || r.density_pct >= 100.0);

    let pass = smallest_ok && degrades && harmonic_scales && exact_only_at_saturation;

    println!();
    println!("  Smallest config fully clean: {}", smallest_ok);
    println!("  System degrades at higher density: {}", degrades);
    println!("  Resolution harmonic increases with density: {}", harmonic_scales);
    println!("  Exact match only fails at 100%% saturation: {}", exact_only_at_saturation);
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}
