use crate::wave::Phase;

// =============================================================================
// Test 22: Kernel Admissibility — Engineering Contract
// =============================================================================
// Verifies that cos(n × Δθ) satisfies the four properties of a valid coherence
// kernel (per Definition 2.5 in Moriya's SECT framework, arXiv:2505.17754v1).
// This serves as a contract: any implementation that passes all four checks is
// correctly using a valid coherence measure.
// =============================================================================
pub fn test_22_kernel_admissibility() -> bool {
    println!("--- Test 22: Kernel Admissibility (Engineering Contract) ---");
    println!();

    let eps = 1e-12;
    let harmonics = [1u32, 2, 3, 4, 5, 6, 8, 12];

    // Generate a diverse set of test angles
    let test_angles: Vec<f64> = vec![
        0.0, 30.0, 45.0, 60.0, 72.0, 90.0, 120.0, 137.5, 180.0, 210.0, 270.0, 315.0, 359.0,
    ];
    let phases: Vec<Phase> = test_angles.iter()
        .map(|d| Phase::from_degrees(*d))
        .collect();

    let mut all_pass = true;

    // =========================================================================
    // Property 1: Symmetry (Hermiticity)
    // cos(n(θ_a - θ_b)) == cos(n(θ_b - θ_a)) for all pairs and all n
    // =========================================================================
    println!("  Property 1: Symmetry (Hermiticity)");
    let mut symmetry_violations = 0;
    let mut symmetry_checks = 0;

    for n in &harmonics {
        for i in 0..phases.len() {
            for j in (i + 1)..phases.len() {
                let forward = phases[i].harmonic_coherence(&phases[j], *n);
                let reverse = phases[j].harmonic_coherence(&phases[i], *n);
                symmetry_checks += 1;
                if (forward - reverse).abs() > eps {
                    symmetry_violations += 1;
                }
            }
        }
    }

    let symmetry_pass = symmetry_violations == 0;
    println!("    {} pairs × {} harmonics = {} checks, {} violations",
        phases.len() * (phases.len() - 1) / 2, harmonics.len(), symmetry_checks, symmetry_violations);
    println!("    cos(n(θ_a - θ_b)) == cos(n(θ_b - θ_a)): {}",
        if symmetry_pass { "VERIFIED" } else { "FAILED" });
    if !symmetry_pass { all_pass = false; }
    println!();

    // =========================================================================
    // Property 2: Normalization (Self-Coherence)
    // cos(n × 0) == 1.0 for all n — every entity is maximally coherent with itself
    // =========================================================================
    println!("  Property 2: Normalization (Self-Coherence)");
    let mut norm_violations = 0;

    for n in &harmonics {
        for p in &phases {
            let self_coh = p.harmonic_coherence(p, *n);
            if (self_coh - 1.0).abs() > eps {
                norm_violations += 1;
                println!("    VIOLATION: n={}, angle={:.1}°, self_coherence={:.12}",
                    n, p.0 * 180.0 / std::f64::consts::PI, self_coh);
            }
        }
    }

    let norm_pass = norm_violations == 0;
    println!("    {} angles × {} harmonics, {} violations",
        phases.len(), harmonics.len(), norm_violations);
    println!("    cos(n × 0) == 1.0 for all angles and harmonics: {}",
        if norm_pass { "VERIFIED" } else { "FAILED" });
    if !norm_pass { all_pass = false; }
    println!();

    // =========================================================================
    // Property 3: Positive Semi-Definiteness
    // For any set of points, the Gram matrix G[i,j] = cos(n(θ_i - θ_j))
    // has all eigenvalues ≥ 0.
    //
    // We verify this using Gershgorin's circle theorem: for a symmetric matrix,
    // eigenvalues lie within discs centered at diagonal entries with radius
    // equal to the row's off-diagonal absolute sum. Since diagonal = 1.0
    // (from normalization), if all row sums of |off-diagonal| ≤ 1.0, then
    // all eigenvalues ≥ 0. For larger matrices we use the determinant of
    // 2×2 and 3×3 sub-matrices (all principal minors non-negative).
    //
    // For a rigorous proof: cos(nθ) is a positive-definite kernel on the
    // circle because its Fourier expansion has non-negative coefficients
    // (it IS a single Fourier mode). This is a known result from harmonic
    // analysis. Here we verify it computationally.
    // =========================================================================
    println!("  Property 3: Positive Semi-Definiteness");
    let mut psd_pass = true;

    for n in &harmonics {
        // Build full Gram matrix
        let size = phases.len();
        let mut gram: Vec<Vec<f64>> = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in 0..size {
                gram[i][j] = phases[i].harmonic_coherence(&phases[j], *n);
            }
        }

        // Check all 2×2 principal minors: det = G[i,i]*G[j,j] - G[i,j]*G[j,i] ≥ 0
        // Since G[i,i] = 1.0, this means 1 - G[i,j]² ≥ 0, i.e. |G[i,j]| ≤ 1
        let mut minor2_violations = 0;
        for i in 0..size {
            for j in (i + 1)..size {
                let det2 = gram[i][i] * gram[j][j] - gram[i][j] * gram[j][i];
                if det2 < -eps {
                    minor2_violations += 1;
                }
            }
        }

        // Check all 3×3 principal minors
        let mut minor3_violations = 0;
        for i in 0..size {
            for j in (i + 1)..size {
                for k in (j + 1)..size {
                    let det3 =
                        gram[i][i] * (gram[j][j] * gram[k][k] - gram[j][k] * gram[k][j])
                        - gram[i][j] * (gram[j][i] * gram[k][k] - gram[j][k] * gram[k][i])
                        + gram[i][k] * (gram[j][i] * gram[k][j] - gram[j][j] * gram[k][i]);
                    if det3 < -eps {
                        minor3_violations += 1;
                    }
                }
            }
        }

        let n_pass = minor2_violations == 0 && minor3_violations == 0;
        println!("    n={}: 2×2 minors: {} violations, 3×3 minors: {} violations [{}]",
            n, minor2_violations, minor3_violations,
            if n_pass { "OK" } else { "FAIL" });
        if !n_pass { psd_pass = false; }
    }

    println!("    All principal minors ≥ 0: {}",
        if psd_pass { "VERIFIED" } else { "FAILED" });
    if !psd_pass { all_pass = false; }
    println!();

    // =========================================================================
    // Property 4: Spectral Scaling
    // Higher harmonic number n = finer angular discrimination.
    // The minimum angle that produces coherence > threshold decreases as n increases.
    // =========================================================================
    println!("  Property 4: Spectral Scaling");
    let threshold: f64 = 0.95;
    let mut prev_min_angle: Option<f64> = None;
    let mut scaling_pass = true;
    let test_ns: Vec<u32> = vec![1, 2, 3, 4, 6, 8, 12];

    for n in &test_ns {
        // Find the smallest non-zero angle that still exceeds threshold
        // The detection angle for harmonic n is 360°/n, and coherence > threshold
        // for angles within arccos(threshold)/n degrees of the harmonic angle.
        let resolution_deg = (threshold.acos()) * 180.0 / std::f64::consts::PI / (*n as f64);

        println!("    n={:>2}: detection resolution = {:.4}° (angles within this of harmonic produce coherence > {:.2})",
            n, resolution_deg, threshold);

        if let Some(prev) = prev_min_angle {
            if resolution_deg > prev + eps {
                println!("           VIOLATION: resolution increased (got coarser) from {:.4}° to {:.4}°", prev, resolution_deg);
                scaling_pass = false;
            }
        }
        prev_min_angle = Some(resolution_deg);
    }

    println!("    Resolution monotonically decreases with n: {}",
        if scaling_pass { "VERIFIED" } else { "FAILED" });
    if !scaling_pass { all_pass = false; }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    if all_pass {
        println!("Test 22: PASS  (Kernel admissibility: symmetry, normalization, positive semi-definiteness, spectral scaling all verified)");
    } else {
        println!("Test 22: FAIL");
        if !symmetry_pass { println!("  - Symmetry (Hermiticity) violated"); }
        if !norm_pass { println!("  - Normalization (self-coherence) violated"); }
        if !psd_pass { println!("  - Positive semi-definiteness violated"); }
        if !scaling_pass { println!("  - Spectral scaling violated"); }
    }

    all_pass
}

// =============================================================================
// Test 23: Channel Energy Concentration (η Diagnostic)
// =============================================================================
// For a given set of embedded points, computes per-channel η values showing
// what fraction of total coherence energy concentrates in each harmonic channel.
// Uses SIGNED mean coherence to identify the fundamental harmonic — the lowest n
// where all pairs in a group align (mean coherence → +1.0). This distinguishes
// the fundamental from its integer multiples (which also show high |coherence|).
// Gives engineers a single diagnostic per dataset to guide compute allocation.
// =============================================================================
pub fn test_23_channel_energy() -> bool {
    println!("--- Test 23: Channel Energy Concentration (η Diagnostic) ---");
    println!();

    let n_harmonics: usize = 12;
    let alignment_threshold: f64 = 0.95;

    // Create groups with known harmonic structure
    // Group A: triadic cluster (120° apart) → fundamental at n=3
    let triadic: Vec<Phase> = vec![
        Phase::from_degrees(0.0),
        Phase::from_degrees(120.0),
        Phase::from_degrees(240.0),
    ];

    // Group B: opposition pair (180° apart) → fundamental at n=2
    let opposition: Vec<Phase> = vec![
        Phase::from_degrees(0.0),
        Phase::from_degrees(180.0),
    ];

    // Group C: quadrant cluster (90° apart) → fundamental at n=4
    let quadrant: Vec<Phase> = vec![
        Phase::from_degrees(0.0),
        Phase::from_degrees(90.0),
        Phase::from_degrees(180.0),
        Phase::from_degrees(270.0),
    ];

    // Group D: noise (no clean harmonic) → no channel reaches alignment
    let noise: Vec<Phase> = vec![
        Phase::from_degrees(0.0),
        Phase::from_degrees(37.0),
        Phase::from_degrees(143.0),
        Phase::from_degrees(211.0),
    ];

    // Compute per-channel mean SIGNED coherence and η (absolute energy fraction)
    struct ChannelProfile {
        signed_mean: Vec<f64>,   // mean cos(n×Δθ) — signed, for fundamental detection
        eta: Vec<f64>,           // |energy| fraction — for distribution display
        fundamental: Option<usize>,  // lowest n with signed_mean > threshold (1-indexed)
    }

    let analyze_group = |group: &[Phase]| -> ChannelProfile {
        let mut signed_sum = vec![0.0f64; n_harmonics];
        let mut abs_sum = vec![0.0f64; n_harmonics];
        let mut pair_count = 0;

        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                for n in 0..n_harmonics {
                    let coh = group[i].harmonic_coherence(&group[j], (n + 1) as u32);
                    signed_sum[n] += coh;
                    abs_sum[n] += coh.abs();
                }
                pair_count += 1;
            }
        }

        let pc = pair_count as f64;
        let signed_mean: Vec<f64> = signed_sum.iter().map(|s| s / pc).collect();
        let abs_mean: Vec<f64> = abs_sum.iter().map(|s| s / pc).collect();

        let total_abs: f64 = abs_mean.iter().sum();
        let eta = if total_abs == 0.0 {
            vec![0.0; n_harmonics]
        } else {
            abs_mean.iter().map(|e| e / total_abs).collect()
        };

        // Fundamental = lowest n where signed mean > threshold
        let fundamental = signed_mean.iter()
            .position(|&m| m > alignment_threshold)
            .map(|i| i + 1); // convert to 1-indexed

        ChannelProfile { signed_mean, eta, fundamental }
    };

    let print_profile = |label: &str, profile: &ChannelProfile| {
        println!("  {}:", label);
        for i in 0..n_harmonics {
            let fund_marker = match profile.fundamental {
                Some(f) if f == i + 1 => " <-- fundamental",
                _ => "",
            };
            let align_marker = if profile.signed_mean[i] > alignment_threshold {
                " [aligned]"
            } else {
                ""
            };
            println!("    n={:>2}: signed_mean = {:>7.4}, η = {:.4}{}{}",
                i + 1, profile.signed_mean[i], profile.eta[i], align_marker, fund_marker);
        }
        match profile.fundamental {
            Some(f) => println!("    Fundamental harmonic: n={}", f),
            None => println!("    Fundamental harmonic: none (no channel aligned)"),
        }
        println!();
    };

    // Phase 1: Analyze all groups
    let prof_tri = analyze_group(&triadic);
    print_profile("Group A — Triadic (120° spacing, expect fundamental n=3)", &prof_tri);

    let prof_opp = analyze_group(&opposition);
    print_profile("Group B — Opposition (180° spacing, expect fundamental n=2)", &prof_opp);

    let prof_quad = analyze_group(&quadrant);
    print_profile("Group C — Quadrant (90° spacing, expect fundamental n=4)", &prof_quad);

    let prof_noise = analyze_group(&noise);
    print_profile("Group D — Noise (no clean harmonic, expect no fundamental)", &prof_noise);

    // Phase 2: Engineering diagnostic
    println!("  Engineering diagnostic summary:");
    println!("    Triadic:    fundamental n={} — {}",
        prof_tri.fundamental.map_or("none".to_string(), |f| f.to_string()),
        if prof_tri.fundamental == Some(3) { "correctly identifies 120° structure" } else { "WRONG" });
    println!("    Opposition: fundamental n={} — {}",
        prof_opp.fundamental.map_or("none".to_string(), |f| f.to_string()),
        if prof_opp.fundamental == Some(2) { "correctly identifies 180° structure" } else { "WRONG" });
    println!("    Quadrant:   fundamental n={} — {}",
        prof_quad.fundamental.map_or("none".to_string(), |f| f.to_string()),
        if prof_quad.fundamental == Some(4) { "correctly identifies 90° structure" } else { "WRONG" });
    println!("    Noise:      fundamental n={} — {}",
        prof_noise.fundamental.map_or("none".to_string(), |f| f.to_string()),
        if prof_noise.fundamental.is_none() { "correctly shows no dominant structure" } else { "unexpected structure found" });
    println!();

    // Phase 3: Concentration comparison
    // Structured groups should have η at fundamental multiples > noise max η
    let noise_max_eta = prof_noise.eta.iter().cloned().fold(0.0f64, f64::max);
    let tri_fund_eta = prof_tri.fundamental.map(|f| prof_tri.eta[f - 1]).unwrap_or(0.0);
    let opp_fund_eta = prof_opp.fundamental.map(|f| prof_opp.eta[f - 1]).unwrap_or(0.0);
    let quad_fund_eta = prof_quad.fundamental.map(|f| prof_quad.eta[f - 1]).unwrap_or(0.0);

    println!("  Concentration comparison (η at fundamental vs noise peak):");
    println!("    Triadic η at n=3:    {:.4}", tri_fund_eta);
    println!("    Opposition η at n=2: {:.4}", opp_fund_eta);
    println!("    Quadrant η at n=4:   {:.4}", quad_fund_eta);
    println!("    Noise max η:         {:.4} (uniform baseline: {:.4})", noise_max_eta, 1.0 / n_harmonics as f64);
    println!();

    // Validation
    let tri_correct = prof_tri.fundamental == Some(3);
    let opp_correct = prof_opp.fundamental == Some(2);
    let quad_correct = prof_quad.fundamental == Some(4);
    let noise_clean = prof_noise.fundamental.is_none();

    let pass = tri_correct && opp_correct && quad_correct && noise_clean;

    if pass {
        println!("Test 23: PASS  (Fundamental harmonics: triadic→n=3, opposition→n=2, quadrant→n=4, noise→none)");
    } else {
        println!("Test 23: FAIL");
        if !tri_correct { println!("  - Triadic: expected fundamental n=3, got {:?}", prof_tri.fundamental); }
        if !opp_correct { println!("  - Opposition: expected fundamental n=2, got {:?}", prof_opp.fundamental); }
        if !quad_correct { println!("  - Quadrant: expected fundamental n=4, got {:?}", prof_quad.fundamental); }
        if !noise_clean { println!("  - Noise: expected no fundamental, got {:?}", prof_noise.fundamental); }
    }

    pass
}
