mod wave;
mod field;
mod relationships;
mod tests;

use tests::*;

fn main() {
    println!("=== Wave Mechanics Test Program ===\n");

    let mut passed = 0;
    let mut failed = 0;

    if test_1_exact_match() { passed += 1; } else { failed += 1; }
    if test_2_harmonic_family() { passed += 1; } else { failed += 1; }
    if test_3_opposition() { passed += 1; } else { failed += 1; }
    if test_4_fuzzy_matching() { passed += 1; } else { failed += 1; }
    if test_5_multi_attribute() { passed += 1; } else { failed += 1; }
    if test_6_directed_cycle() { passed += 1; } else { failed += 1; }
    if test_7_structural_pairs() { passed += 1; } else { failed += 1; }
    if test_8_wave_vs_linear() { passed += 1; } else { failed += 1; }
    if test_9_harmonic_vs_join() { passed += 1; } else { failed += 1; }
    if test_10_typed_reach() { passed += 1; } else { failed += 1; }

    println!("\n=== EXPERIMENTAL TESTS ===\n");
    if test_11_harmonic_fingerprint() { passed += 1; } else { failed += 1; }
    if test_12_mutual_amplification() { passed += 1; } else { failed += 1; }
    if test_13_cycle_uniqueness() { passed += 1; } else { failed += 1; }
    if test_14_harmonic_orthogonality() { passed += 1; } else { failed += 1; }
    if test_15_wraparound() { passed += 1; } else { failed += 1; }
    if test_16_scale_resolution() { passed += 1; } else { failed += 1; }
    if test_17_scaling_limits() { passed += 1; } else { failed += 1; }
    if test_18_bucket_index() { passed += 1; } else { failed += 1; }

    let total = passed + failed;
    println!("\n=== RESULTS: {passed} passed, {failed} failed out of {total} ===");
    if failed == 0 {
        println!("ALL TESTS PASSED.");
    } else {
        println!("SOME TESTS FAILED — review output above.");
    }
}
