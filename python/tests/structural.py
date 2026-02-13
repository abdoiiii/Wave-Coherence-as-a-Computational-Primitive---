"""Tests 6-7: Directed cycles and structural pairs."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wave import Phase
from relationships import DirectedCycle, PairTable


def test_6_directed_cycle() -> bool:
    print("--- Test 6: Directed Cycle Traversal ---")
    cycle = DirectedCycle(5)

    gen_chain = cycle.chain(0, 1, 3)
    print(f"  Generative from 0, depth 3: {gen_chain}")
    gen_pass = gen_chain == [0, 1, 2, 3]

    dest_chain = cycle.chain(0, 2, 3)
    print(f"  Destructive from 0, depth 3: {dest_chain}")
    dest_pass = dest_chain == [0, 2, 4, 1]

    gen_chain_2 = cycle.chain(3, 1, 2)
    print(f"  Generative from 3, depth 2: {gen_chain_2}")
    gen2_pass = gen_chain_2 == [3, 4, 0]

    weak_chain = cycle.chain(0, -1, 4)
    print(f"  Weakening from 0, depth 4:  {weak_chain}")
    weak_pass = weak_chain == [0, 4, 3, 2, 1]

    ctrl_chain = cycle.chain(0, -2, 4)
    print(f"  Controlling from 0, depth 4: {ctrl_chain}")
    ctrl_pass = ctrl_chain == [0, 3, 1, 4, 2]

    passed = gen_pass and dest_pass and gen2_pass and weak_pass and ctrl_pass
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_7_structural_pairs() -> bool:
    print("--- Test 7: Structural Pair Lookup ---")
    table = PairTable([(0, 1), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)])

    all_pass = True

    p0 = table.partner(0)
    print(f"  Position 0's partner: {p0} (expected: 1)")
    all_pass = all_pass and p0 == 1

    p2 = table.partner(2)
    print(f"  Position 2's partner: {p2} (expected: 11)")
    all_pass = all_pass and p2 == 11

    p3 = table.partner(3)
    print(f"  Position 3's partner: {p3} (expected: 10)")
    all_pass = all_pass and p3 == 10

    paired_0_4 = table.are_paired(0, 4)
    print(f"  0 paired with 4 (120 deg harmonic): {paired_0_4} (expected: False)")
    all_pass = all_pass and not paired_0_4

    ring_size = 12
    print("  Angular distances of structural pairs (on 12-position ring):")
    for a, b in [(0, 1), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]:
        angle_a = Phase.from_value(a, ring_size)
        angle_b = Phase.from_value(b, ring_size)
        print(f"    ({a}, {b}) -> {angle_a.distance_degrees(angle_b):.1f} deg")

    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass
