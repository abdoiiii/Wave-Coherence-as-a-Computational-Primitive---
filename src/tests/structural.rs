use crate::wave::Phase;
use crate::relationships::{DirectedCycle, PairTable};

// =============================================================================
// Test 6: Directed Cycle Traversal
// =============================================================================
pub fn test_6_directed_cycle() -> bool {
    println!("--- Test 6: Directed Cycle Traversal ---");
    let cycle = DirectedCycle::new(5);

    // From position 0, generative (+1) chain depth 3: [0, 1, 2, 3]
    let gen_chain = cycle.chain(0, 1, 3);
    println!("  Generative from 0, depth 3: {:?}", gen_chain);
    let gen_pass = gen_chain == vec![0, 1, 2, 3];

    // From position 0, destructive (+2) chain depth 3: [0, 2, 4, 1]
    let dest_chain = cycle.chain(0, 2, 3);
    println!("  Destructive from 0, depth 3: {:?}", dest_chain);
    let dest_pass = dest_chain == vec![0, 2, 4, 1];

    // From position 3, generative (+1) chain depth 2: [3, 4, 0]
    let gen_chain_2 = cycle.chain(3, 1, 2);
    println!("  Generative from 3, depth 2: {:?}", gen_chain_2);
    let gen2_pass = gen_chain_2 == vec![3, 4, 0];

    // Also verify weakening (-1) and controlling (-2)
    let weak_chain = cycle.chain(0, -1, 4);
    println!("  Weakening from 0, depth 4:  {:?}", weak_chain);
    let weak_pass = weak_chain == vec![0, 4, 3, 2, 1];

    let ctrl_chain = cycle.chain(0, -2, 4);
    println!("  Controlling from 0, depth 4: {:?}", ctrl_chain);
    let ctrl_pass = ctrl_chain == vec![0, 3, 1, 4, 2];

    let pass = gen_pass && dest_pass && gen2_pass && weak_pass && ctrl_pass;
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// =============================================================================
// Test 7: Structural Pair Lookup
// =============================================================================
pub fn test_7_structural_pairs() -> bool {
    println!("--- Test 7: Structural Pair Lookup ---");
    let table = PairTable::new(vec![
        (0, 1), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7),
    ]);

    let mut all_pass = true;

    // Position 0's partner = 1
    let p0 = table.partner(0);
    println!("  Position 0's partner: {:?} (expected: Some(1))", p0);
    all_pass &= p0 == Some(1);

    // Position 2's partner = 11
    let p2 = table.partner(2);
    println!("  Position 2's partner: {:?} (expected: Some(11))", p2);
    all_pass &= p2 == Some(11);

    // Position 3's partner = 10
    let p3 = table.partner(3);
    println!("  Position 3's partner: {:?} (expected: Some(10))", p3);
    all_pass &= p3 == Some(10);

    // Position 0 is NOT paired with position 4 (even though 120° is a harmonic)
    let paired_0_4 = table.are_paired(0, 4);
    println!("  0 paired with 4 (120° harmonic): {} (expected: false)", paired_0_4);
    all_pass &= !paired_0_4;

    // Verify angular distances to show these are NON-geometric
    let ring_size = 12;
    println!("  Angular distances of structural pairs (on 12-position ring):");
    for &(a, b) in &[(0usize, 1usize), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)] {
        let angle_a = Phase::from_value(a as u64, ring_size);
        let angle_b = Phase::from_value(b as u64, ring_size);
        println!("    ({}, {}) -> {:.1}°", a, b, angle_a.distance_degrees(&angle_b));
    }

    println!("  RESULT: {}\n", if all_pass { "PASS" } else { "FAIL" });
    all_pass
}
