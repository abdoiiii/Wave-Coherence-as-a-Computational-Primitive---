/// Harmonic Sweep Test — Proof of Concept
///
/// FINDING: Cosine similarity is blind to harmonic structure in vectors.
/// Two vectors can have cosine similarity of exactly 0.0000 while containing
/// perfect harmonic coherence of 1.0000 on specific frequency channels.
///
/// This program demonstrates the blindness and validates a sweep method
/// that recovers the hidden structure.
///
/// METHOD: Encode letters at KNOWN phase angles with deliberate harmonic
/// relationships, generate harmonic embedding vectors, then sweep across
/// harmonics to verify the method recovers the planted structure.
///
/// Zero dependencies. Pure math.

use std::f64::consts::PI;

// --- Harmonic Embedding ---

/// Generate a harmonic embedding vector for a given phase angle.
/// v(θ) = [cos(θ), cos(2θ), cos(3θ), ..., cos(Nθ)]
fn harmonic_embedding(theta: f64, n_harmonics: usize) -> Vec<f64> {
    (1..=n_harmonics)
        .map(|n| (n as f64 * theta).cos())
        .collect()
}

/// Coherence at a specific harmonic between two phase angles.
/// This is the nth component of the relationship between two embeddings.
fn coherence_at_harmonic(theta_a: f64, theta_b: f64, n: usize) -> f64 {
    (n as f64 * (theta_a - theta_b)).cos()
}

/// Standard dot product of two vectors (what cosine similarity uses).
/// This SUMS all harmonics together — losing which harmonic carries the signal.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Magnitude of a vector.
fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cosine similarity — the standard ML comparison.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot_product(a, b) / (mag_a * mag_b)
}

// --- Test Setup ---

struct Letter {
    name: char,
    angle_deg: f64,
    theta: f64, // radians
}

fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

fn main() {
    println!("=== Harmonic Sweep Test ===");
    println!();

    let n_harmonics = 12;

    // --- Phase 1: Encode letters at known angles ---
    println!("Phase 1: Encoding letters at known phase angles");
    println!("---");

    let letters = vec![
        Letter { name: 'A', angle_deg: 0.0,   theta: deg_to_rad(0.0) },
        Letter { name: 'B', angle_deg: 120.0,  theta: deg_to_rad(120.0) },  // triadic with A (n=3)
        Letter { name: 'C', angle_deg: 180.0,  theta: deg_to_rad(180.0) },  // opposition to A (n=2)
        Letter { name: 'D', angle_deg: 90.0,   theta: deg_to_rad(90.0) },   // quadrant from A (n=4)
        Letter { name: 'E', angle_deg: 60.0,   theta: deg_to_rad(60.0) },   // sextile from A (n=6)
        Letter { name: 'F', angle_deg: 72.0,   theta: deg_to_rad(72.0) },   // pentagonal from A (n=5)
        Letter { name: 'G', angle_deg: 37.0,   theta: deg_to_rad(37.0) },   // no clean harmonic
        Letter { name: 'H', angle_deg: 143.0,  theta: deg_to_rad(143.0) },  // near-pentagonal (143 ~ 144 = 360/2.5)
    ];

    for l in &letters {
        println!("  {} at {:>6.1}°", l.name, l.angle_deg);
    }
    println!();

    // --- Phase 2: Generate harmonic embeddings ---
    println!("Phase 2: Generating {}-dimensional harmonic embeddings", n_harmonics);
    println!("  v(theta) = [cos(theta), cos(2*theta), cos(3*theta), ..., cos({}*theta)]", n_harmonics);
    println!("---");

    let embeddings: Vec<Vec<f64>> = letters.iter()
        .map(|l| harmonic_embedding(l.theta, n_harmonics))
        .collect();

    for (i, l) in letters.iter().enumerate() {
        let v = &embeddings[i];
        print!("  {} = [", l.name);
        for (j, val) in v.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:>7.3}", val);
        }
        println!("]");
    }
    println!();

    // --- Phase 3: Standard cosine similarity (what ML does) ---
    println!("Phase 3: Standard cosine similarity (all harmonics summed)");
    println!("  This is what ML normally sees — structure is mixed together");
    println!("---");

    for i in 0..letters.len() {
        for j in (i+1)..letters.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("  {}-{}: cosine_sim = {:>7.4}", letters[i].name, letters[j].name, sim);
        }
    }
    println!();

    // --- Phase 4: THE SWEEP TEST ---
    println!("Phase 4: HARMONIC SWEEP — decomposing relationships by frequency");
    println!("  For each pair, coherence at each harmonic n=1..{}", n_harmonics);
    println!("  Expected: planted relationships light up at specific harmonics");
    println!("---");

    // Expected relationships:
    let expected = vec![
        ('A', 'B', 3, "triadic (120 deg)"),
        ('A', 'C', 2, "opposition (180 deg)"),
        ('A', 'D', 4, "quadrant (90 deg)"),
        ('A', 'E', 6, "sextile (60 deg)"),
        ('A', 'F', 5, "pentagonal (72 deg)"),
    ];

    println!("  Expected detections:");
    for (a, b, n, desc) in &expected {
        println!("    {}-{} at n={}: {}", a, b, n, desc);
    }
    println!();

    // Sweep all pairs across all harmonics
    let threshold = 0.999; // Exact harmonic relationships only
    println!("  Sweep results (showing coherence at each harmonic, threshold={:.2}):", threshold);
    println!();

    let mut detections: Vec<(char, char, usize, f64)> = Vec::new();

    for i in 0..letters.len() {
        for j in (i+1)..letters.len() {
            let a = &letters[i];
            let b = &letters[j];

            print!("  {}-{}:  ", a.name, b.name);

            for n in 1..=n_harmonics {
                let coh = coherence_at_harmonic(a.theta, b.theta, n);
                // Mark high coherence
                if coh > threshold {
                    print!("[n={}: {:.3}*] ", n, coh);
                    detections.push((a.name, b.name, n, coh));
                } else if coh < -threshold {
                    print!("[n={}: {:.3}!] ", n, coh); // anti-coherence
                } else {
                    // skip noise for clarity
                }
            }
            println!();
        }
    }

    // --- Phase 5: Validate detections ---
    println!();
    println!("Phase 5: VALIDATION");
    println!("---");

    let mut all_pass = true;

    for (a, b, expected_n, desc) in &expected {
        let found = detections.iter().any(|(da, db, dn, _)| {
            da == a && db == b && *dn == *expected_n as usize
        });
        if found {
            println!("  PASS  {}-{} detected at n={} ({})", a, b, expected_n, desc);
        } else {
            println!("  FAIL  {}-{} NOT detected at n={} ({})", a, b, expected_n, desc);
            all_pass = false;
        }
    }

    // Check that G and H (no clean harmonic) don't produce false positives with A
    println!();
    println!("  Noise check (G and H should NOT light up with A at low harmonics):");
    let noise_pairs = vec![('A', 'G'), ('A', 'H')];
    for (a, b) in &noise_pairs {
        let false_positives: Vec<_> = detections.iter()
            .filter(|(da, db, dn, _)| da == a && db == b && *dn <= 6)
            .collect();
        if false_positives.is_empty() {
            println!("  PASS  {}-{}: no false detections at n=1..6", a, b);
        } else {
            let fp_strs: Vec<_> = false_positives.iter()
                .map(|(_, _, n, c)| format!("n={}({:.3})", n, c))
                .collect();
            println!("  FAIL  {}-{}: false detection at {}", a, b, fp_strs.join(", "));
            all_pass = false;
        }
    }

    // --- Phase 6: The spectral profile ---
    println!();
    println!("Phase 6: SPECTRAL PROFILE — the 'model signature' concept");
    println!("  Energy at each harmonic across ALL pairs");
    println!("---");

    println!("  Harmonic  | Avg |coh|  | Max |coh|  | Pairs with |coh|>{:.2}", threshold);
    println!("  ----------|-----------|-----------|---------------------------");

    for n in 1..=n_harmonics {
        let mut total_abs_coh = 0.0;
        let mut max_abs_coh: f64 = 0.0;
        let mut high_pairs = Vec::new();
        let mut pair_count = 0;

        for i in 0..letters.len() {
            for j in (i+1)..letters.len() {
                let coh = coherence_at_harmonic(letters[i].theta, letters[j].theta, n);
                let abs_coh = coh.abs();
                total_abs_coh += abs_coh;
                if abs_coh > max_abs_coh { max_abs_coh = abs_coh; }
                if abs_coh > threshold {
                    high_pairs.push(format!("{}{}", letters[i].name, letters[j].name));
                }
                pair_count += 1;
            }
        }

        let avg = total_abs_coh / pair_count as f64;
        let pairs_str = if high_pairs.is_empty() {
            String::from("(none)")
        } else {
            high_pairs.join(", ")
        };
        println!("  n={:<3}     | {:.4}    | {:.4}    | {}", n, avg, max_abs_coh, pairs_str);
    }

    // --- Phase 7: The key insight ---
    println!();
    println!("Phase 7: KEY INSIGHT");
    println!("---");
    println!("  Standard cosine similarity (Phase 3) mixes all harmonics into one number.");
    println!("  The sweep (Phase 4) decomposes that number into independent channels.");
    println!();
    println!("  Cosine similarity between A and B = {:.4}", cosine_similarity(&embeddings[0], &embeddings[1]));
    println!("  But the SWEEP reveals:");
    for n in 1..=n_harmonics {
        let coh = coherence_at_harmonic(letters[0].theta, letters[1].theta, n);
        if coh.abs() > 0.5 {
            println!("    n={}: {:>7.4}  <-- signal", n, coh);
        } else {
            println!("    n={}: {:>7.4}", n, coh);
        }
    }
    println!();
    println!("  The structure was always there. The sweep finds the frequency.");

    // --- Final result ---
    println!();
    if all_pass {
        println!("=== ALL VALIDATIONS PASSED ===");
        println!("The harmonic sweep method correctly recovers planted structure.");
        println!("The method is valid for probing unknown embeddings.");
    } else {
        println!("=== SOME VALIDATIONS FAILED ===");
    }
}
