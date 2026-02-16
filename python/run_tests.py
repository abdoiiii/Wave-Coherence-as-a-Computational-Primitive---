#!/usr/bin/env python3
"""
Wave Mechanics Test Program — Python translation.

Runs all 21 validation tests for wave coherence as a computational primitive.
Produces output identical to the Rust version (cargo run).

Requirements: Python 3.10+ (uses math only — zero external dependencies).

Usage:
    python run_tests.py
"""

import sys
import os

# Ensure imports work regardless of how the script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.core_tests import (
    test_1_exact_match,
    test_2_harmonic_family,
    test_3_opposition,
    test_4_fuzzy_matching,
    test_5_multi_attribute,
)
from tests.structural import (
    test_6_directed_cycle,
    test_7_structural_pairs,
)
from tests.comparison import (
    test_8_wave_vs_linear,
    test_9_harmonic_vs_join,
)
from tests.advanced import (
    test_10_typed_reach,
    test_11_harmonic_fingerprint,
    test_12_mutual_amplification,
    test_13_cycle_uniqueness,
)
from tests.boundary import (
    test_14_harmonic_orthogonality,
    test_15_wraparound,
    test_16_scale_resolution,
)
from tests.scaling import (
    test_17_scaling_limits,
)
from tests.indexing import (
    test_18_bucket_index,
    test_19_multi_attr_index,
    test_20_dynamic_mutation,
)
from tests.sweep import (
    test_21_harmonic_sweep,
)
from tests.kernel import (
    test_22_kernel_admissibility,
    test_23_channel_energy,
)


def main():
    print("=== Wave Mechanics Test Program ===\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_1_exact_match,
        test_2_harmonic_family,
        test_3_opposition,
        test_4_fuzzy_matching,
        test_5_multi_attribute,
        test_6_directed_cycle,
        test_7_structural_pairs,
        test_8_wave_vs_linear,
        test_9_harmonic_vs_join,
        test_10_typed_reach,
    ]:
        if test_fn():
            passed += 1
        else:
            failed += 1

    print("\n=== EXPERIMENTAL TESTS ===\n")

    for test_fn in [
        test_11_harmonic_fingerprint,
        test_12_mutual_amplification,
        test_13_cycle_uniqueness,
        test_14_harmonic_orthogonality,
        test_15_wraparound,
        test_16_scale_resolution,
        test_17_scaling_limits,
        test_18_bucket_index,
        test_19_multi_attr_index,
        test_20_dynamic_mutation,
        test_21_harmonic_sweep,
        test_22_kernel_admissibility,
        test_23_channel_energy,
    ]:
        if test_fn():
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n=== RESULTS: {passed} passed, {failed} failed out of {total} ===")
    if failed == 0:
        print("ALL TESTS PASSED.")
    else:
        print("SOME TESTS FAILED — review output above.")


if __name__ == "__main__":
    main()
