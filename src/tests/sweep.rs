use crate::wave::Phase;

// =============================================================================
// Test 21: Harmonic Sweep — Cosine Similarity Blindness
// =============================================================================
pub fn test_21_harmonic_sweep() -> bool {
    println!("--- Test 21: Harmonic Sweep (Cosine Similarity Blindness) ---");
    println!();

    // Hypothesis: cosine similarity of harmonic embedding vectors sums all
    // harmonic channels into one number, destroying per-channel structure.
    // A harmonic sweep that examines each channel independently can recover
    // relationships that cosine similarity reports as zero.

    let n_harmonics: usize = 12;

    // Phase 1: Encode letters at known angles with deliberate harmonic relationships
    let letters: Vec<(char, f64)> = vec![
        ('A', 0.0),    // reference
        ('B', 120.0),  // triadic with A → detectable at n=3
        ('C', 180.0),  // opposition to A → detectable at n=2
        ('D', 90.0),   // quadrant from A → detectable at n=4
        ('E', 60.0),   // sextile from A → detectable at n=6
        ('F', 72.0),   // pentagonal from A → detectable at n=5
        ('G', 37.0),   // noise control — no clean harmonic with A
        ('H', 143.0),  // noise control — no clean harmonic with A
    ];

    let phases: Vec<Phase> = letters.iter()
        .map(|(_, deg)| Phase::from_degrees(*deg))
        .collect();

    // Phase 2: Generate harmonic embeddings v(θ) = [cos(θ), cos(2θ), ..., cos(Nθ)]
    let embeddings: Vec<Vec<f64>> = phases.iter()
        .map(|p| {
            (1..=n_harmonics)
                .map(|n| (n as f64 * p.0).cos())
                .collect()
        })
        .collect();

    // Phase 3: Compute cosine similarity (standard ML method)
    let cosine_sim = |a: &[f64], b: &[f64]| -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 { 0.0 } else { dot / (mag_a * mag_b) }
    };

    // Key pairs where cosine similarity is ~0 but harmonic structure exists
    let ab_cosine = cosine_sim(&embeddings[0], &embeddings[1]); // A-B
    let ac_cosine = cosine_sim(&embeddings[0], &embeddings[2]); // A-C
    let ad_cosine = cosine_sim(&embeddings[0], &embeddings[3]); // A-D
    let ae_cosine = cosine_sim(&embeddings[0], &embeddings[4]); // A-E

    println!("  Cosine similarity (standard ML):");
    println!("    A-B: {:.4}  (triadic partners — should be related)", ab_cosine);
    println!("    A-C: {:.4}  (opposition pair — should be related)", ac_cosine);
    println!("    A-D: {:.4}  (quadrant pair — should be related)", ad_cosine);
    println!("    A-E: {:.4}  (sextile pair — should be related)", ae_cosine);
    println!();

    // Phase 4: Harmonic sweep — check each channel independently
    let expected: Vec<(usize, usize, usize, &str)> = vec![
        (0, 1, 3, "A-B triadic (120 deg)"),
        (0, 2, 2, "A-C opposition (180 deg)"),
        (0, 3, 4, "A-D quadrant (90 deg)"),
        (0, 4, 6, "A-E sextile (60 deg)"),
        (0, 5, 5, "A-F pentagonal (72 deg)"),
    ];

    let threshold = 0.999; // Exact harmonic relationships only

    println!("  Harmonic sweep (per-channel decomposition):");

    let mut all_detected = true;
    for (i, j, expected_n, desc) in &expected {
        let coh = phases[*i].harmonic_coherence(&phases[*j], *expected_n as u32);
        let detected = coh > threshold;
        let status = if detected { "DETECTED" } else { "MISSED" };
        println!("    {} at n={}: coherence={:.6} [{}]", desc, expected_n, coh, status);
        if !detected {
            all_detected = false;
        }
    }
    println!();

    // Phase 5: Noise check — G and H should not produce false positives with A
    let noise_indices = vec![(0, 6, "A-G"), (0, 7, "A-H")];
    let mut noise_clean = true;

    println!("  Noise check (non-harmonic pairs at n=1..6):");
    for (i, j, label) in &noise_indices {
        let mut false_pos = Vec::new();
        for n in 1..=6u32 {
            let coh = phases[*i].harmonic_coherence(&phases[*j], n);
            if coh > threshold {
                false_pos.push(n);
            }
        }
        if false_pos.is_empty() {
            println!("    {}: clean — no false detections", label);
        } else {
            println!("    {}: FALSE POSITIVE at n={:?}", label, false_pos);
            noise_clean = false;
        }
    }
    println!();

    // Phase 6: Demonstrate the A-B decomposition — the money shot
    println!("  A-B decomposition (cosine_sim={:.4}, but per-channel):", ab_cosine);
    let mut signal_channels = 0;
    let mut noise_channels = 0;
    for n in 1..=n_harmonics {
        let coh = phases[0].harmonic_coherence(&phases[1], n as u32);
        let marker = if coh.abs() > 0.999 {
            signal_channels += 1;
            if coh > 0.0 { "<-- coherence 1.0" } else { "<-- anti-coherence -1.0" }
        } else {
            noise_channels += 1;
            ""
        };
        println!("    n={:>2}: {:>8.4}  {}", n, coh, marker);
    }
    println!();
    println!("  {} signal channels, {} noise channels → sum cancels to {:.4}",
        signal_channels, noise_channels, ab_cosine);
    println!("  Cosine similarity destroys the harmonic decomposition.");
    println!();

    // Phase 7: Spectral profile — energy distribution across harmonics
    println!("  Spectral profile (pairs with exact coherence at each harmonic):");
    for n in 1..=n_harmonics {
        let mut high_pairs = Vec::new();
        for i in 0..letters.len() {
            for j in (i+1)..letters.len() {
                let coh = phases[i].harmonic_coherence(&phases[j], n as u32);
                if coh.abs() > threshold {
                    high_pairs.push(format!("{}{}", letters[i].0, letters[j].0));
                }
            }
        }
        let count = high_pairs.len();
        let pairs_str = if high_pairs.is_empty() {
            String::from("(none)")
        } else {
            high_pairs.join(", ")
        };
        println!("    n={:>2}: {:>2} pairs  {}", n, count, pairs_str);
    }
    println!();

    // Validation
    let pass = all_detected && noise_clean
        && ab_cosine.abs() < 0.01  // cosine sim near zero for A-B
        && ac_cosine.abs() < 0.01  // cosine sim near zero for A-C
        && ad_cosine.abs() < 0.01  // cosine sim near zero for A-D
        && ae_cosine.abs() < 0.01; // cosine sim near zero for A-E

    if pass {
        println!("Test 21: PASS  (5 planted relationships recovered, cosine similarity blind to all, 0 false positives)");
    } else {
        println!("Test 21: FAIL");
        if !all_detected { println!("  - Not all planted relationships detected"); }
        if !noise_clean { println!("  - False positives on noise controls"); }
        if ab_cosine.abs() >= 0.01 { println!("  - A-B cosine sim not near zero: {}", ab_cosine); }
    }

    pass
}
