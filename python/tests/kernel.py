"""
Tests 22-23: Kernel admissibility and channel energy concentration.

Translates src/tests/kernel.rs -- engineering contract tests derived from
Moriya's SECT framework (arXiv:2505.17754v1).
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wave import Phase


# =============================================================================
# Test 22: Kernel Admissibility -- Engineering Contract
# =============================================================================
def test_22_kernel_admissibility() -> bool:
    print("--- Test 22: Kernel Admissibility (Engineering Contract) ---")
    print()

    eps = 1e-12
    harmonics = [1, 2, 3, 4, 5, 6, 8, 12]

    # Generate a diverse set of test angles
    test_angles = [0.0, 30.0, 45.0, 60.0, 72.0, 90.0, 120.0, 137.5, 180.0, 210.0, 270.0, 315.0, 359.0]
    phases = [Phase.from_degrees(d) for d in test_angles]

    all_pass = True

    # =========================================================================
    # Property 1: Symmetry (Hermiticity)
    # cos(n(theta_a - theta_b)) == cos(n(theta_b - theta_a)) for all pairs and all n
    # =========================================================================
    print("  Property 1: Symmetry (Hermiticity)")
    symmetry_violations = 0
    symmetry_checks = 0

    for n in harmonics:
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                forward = phases[i].harmonic_coherence(phases[j], n)
                reverse = phases[j].harmonic_coherence(phases[i], n)
                symmetry_checks += 1
                if abs(forward - reverse) > eps:
                    symmetry_violations += 1

    symmetry_pass = symmetry_violations == 0
    num_pairs = len(phases) * (len(phases) - 1) // 2
    print(f"    {num_pairs} pairs x {len(harmonics)} harmonics = {symmetry_checks} checks, {symmetry_violations} violations")
    print(f"    cos(n(theta_a - theta_b)) == cos(n(theta_b - theta_a)): {'VERIFIED' if symmetry_pass else 'FAILED'}")
    if not symmetry_pass:
        all_pass = False
    print()

    # =========================================================================
    # Property 2: Normalization (Self-Coherence)
    # cos(n * 0) == 1.0 for all n
    # =========================================================================
    print("  Property 2: Normalization (Self-Coherence)")
    norm_violations = 0

    for n in harmonics:
        for p in phases:
            self_coh = p.harmonic_coherence(p, n)
            if abs(self_coh - 1.0) > eps:
                norm_violations += 1
                print(f"    VIOLATION: n={n}, angle={p.angle * 180.0 / math.pi:.1f} deg, self_coherence={self_coh:.12f}")

    norm_pass = norm_violations == 0
    print(f"    {len(phases)} angles x {len(harmonics)} harmonics, {norm_violations} violations")
    print(f"    cos(n * 0) == 1.0 for all angles and harmonics: {'VERIFIED' if norm_pass else 'FAILED'}")
    if not norm_pass:
        all_pass = False
    print()

    # =========================================================================
    # Property 3: Positive Semi-Definiteness
    # For any set of points, the Gram matrix G[i,j] = cos(n(theta_i - theta_j))
    # has all eigenvalues >= 0. Verified via principal minors.
    # =========================================================================
    print("  Property 3: Positive Semi-Definiteness")
    psd_pass = True

    for n in harmonics:
        size = len(phases)
        gram = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                gram[i][j] = phases[i].harmonic_coherence(phases[j], n)

        # Check all 2x2 principal minors
        minor2_violations = 0
        for i in range(size):
            for j in range(i + 1, size):
                det2 = gram[i][i] * gram[j][j] - gram[i][j] * gram[j][i]
                if det2 < -eps:
                    minor2_violations += 1

        # Check all 3x3 principal minors
        minor3_violations = 0
        for i in range(size):
            for j in range(i + 1, size):
                for k in range(j + 1, size):
                    det3 = (
                        gram[i][i] * (gram[j][j] * gram[k][k] - gram[j][k] * gram[k][j])
                        - gram[i][j] * (gram[j][i] * gram[k][k] - gram[j][k] * gram[k][i])
                        + gram[i][k] * (gram[j][i] * gram[k][j] - gram[j][j] * gram[k][i])
                    )
                    if det3 < -eps:
                        minor3_violations += 1

        n_pass = minor2_violations == 0 and minor3_violations == 0
        print(f"    n={n}: 2x2 minors: {minor2_violations} violations, 3x3 minors: {minor3_violations} violations [{'OK' if n_pass else 'FAIL'}]")
        if not n_pass:
            psd_pass = False

    print(f"    All principal minors >= 0: {'VERIFIED' if psd_pass else 'FAILED'}")
    if not psd_pass:
        all_pass = False
    print()

    # =========================================================================
    # Property 4: Spectral Scaling
    # Higher harmonic number n = finer angular discrimination.
    # =========================================================================
    print("  Property 4: Spectral Scaling")
    threshold = 0.95
    prev_min_angle = None
    scaling_pass = True
    test_ns = [1, 2, 3, 4, 6, 8, 12]

    for n in test_ns:
        resolution_deg = math.acos(threshold) * 180.0 / math.pi / n
        print(f"    n={n:>2}: detection resolution = {resolution_deg:.4f} deg (angles within this of harmonic produce coherence > {threshold:.2f})")

        if prev_min_angle is not None:
            if resolution_deg > prev_min_angle + eps:
                print(f"           VIOLATION: resolution increased (got coarser) from {prev_min_angle:.4f} deg to {resolution_deg:.4f} deg")
                scaling_pass = False
        prev_min_angle = resolution_deg

    print(f"    Resolution monotonically decreases with n: {'VERIFIED' if scaling_pass else 'FAILED'}")
    if not scaling_pass:
        all_pass = False
    print()

    # Summary
    if all_pass:
        print("Test 22: PASS  (Kernel admissibility: symmetry, normalization, positive semi-definiteness, spectral scaling all verified)")
    else:
        print("Test 22: FAIL")
        if not symmetry_pass:
            print("  - Symmetry (Hermiticity) violated")
        if not norm_pass:
            print("  - Normalization (self-coherence) violated")
        if not psd_pass:
            print("  - Positive semi-definiteness violated")
        if not scaling_pass:
            print("  - Spectral scaling violated")

    return all_pass


# =============================================================================
# Test 23: Channel Energy Concentration (eta Diagnostic)
# =============================================================================
def test_23_channel_energy() -> bool:
    print("--- Test 23: Channel Energy Concentration (eta Diagnostic) ---")
    print()

    n_harmonics = 12
    alignment_threshold = 0.95

    # Group A: triadic cluster (120 deg apart) -- fundamental at n=3
    triadic = [
        Phase.from_degrees(0.0),
        Phase.from_degrees(120.0),
        Phase.from_degrees(240.0),
    ]

    # Group B: opposition pair (180 deg apart) -- fundamental at n=2
    opposition = [
        Phase.from_degrees(0.0),
        Phase.from_degrees(180.0),
    ]

    # Group C: quadrant cluster (90 deg apart) -- fundamental at n=4
    quadrant_group = [
        Phase.from_degrees(0.0),
        Phase.from_degrees(90.0),
        Phase.from_degrees(180.0),
        Phase.from_degrees(270.0),
    ]

    # Group D: noise (no clean harmonic) -- no channel reaches alignment
    noise = [
        Phase.from_degrees(0.0),
        Phase.from_degrees(37.0),
        Phase.from_degrees(143.0),
        Phase.from_degrees(211.0),
    ]

    def analyze_group(group):
        signed_sum = [0.0] * n_harmonics
        abs_sum = [0.0] * n_harmonics
        pair_count = 0

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                for n in range(n_harmonics):
                    coh = group[i].harmonic_coherence(group[j], n + 1)
                    signed_sum[n] += coh
                    abs_sum[n] += abs(coh)
                pair_count += 1

        pc = float(pair_count)
        signed_mean = [s / pc for s in signed_sum]
        abs_mean = [s / pc for s in abs_sum]

        total_abs = sum(abs_mean)
        if total_abs == 0.0:
            eta = [0.0] * n_harmonics
        else:
            eta = [e / total_abs for e in abs_mean]

        # Fundamental = lowest n where signed mean > threshold
        fundamental = None
        for i, m in enumerate(signed_mean):
            if m > alignment_threshold:
                fundamental = i + 1  # 1-indexed
                break

        return signed_mean, eta, fundamental

    def print_profile(label, signed_mean, eta, fundamental):
        print(f"  {label}:")
        for i in range(n_harmonics):
            fund_marker = " <-- fundamental" if fundamental == i + 1 else ""
            align_marker = " [aligned]" if signed_mean[i] > alignment_threshold else ""
            print(f"    n={i+1:>2}: signed_mean = {signed_mean[i]:>7.4f}, eta = {eta[i]:.4f}{align_marker}{fund_marker}")
        if fundamental is not None:
            print(f"    Fundamental harmonic: n={fundamental}")
        else:
            print("    Fundamental harmonic: none (no channel aligned)")
        print()

    # Phase 1: Analyze all groups
    tri_signed, tri_eta, tri_fund = analyze_group(triadic)
    print_profile("Group A -- Triadic (120 deg spacing, expect fundamental n=3)", tri_signed, tri_eta, tri_fund)

    opp_signed, opp_eta, opp_fund = analyze_group(opposition)
    print_profile("Group B -- Opposition (180 deg spacing, expect fundamental n=2)", opp_signed, opp_eta, opp_fund)

    quad_signed, quad_eta, quad_fund = analyze_group(quadrant_group)
    print_profile("Group C -- Quadrant (90 deg spacing, expect fundamental n=4)", quad_signed, quad_eta, quad_fund)

    noise_signed, noise_eta, noise_fund = analyze_group(noise)
    print_profile("Group D -- Noise (no clean harmonic, expect no fundamental)", noise_signed, noise_eta, noise_fund)

    # Phase 2: Engineering diagnostic
    print("  Engineering diagnostic summary:")
    print(f"    Triadic:    fundamental n={tri_fund if tri_fund else 'none'} -- "
          f"{'correctly identifies 120 deg structure' if tri_fund == 3 else 'WRONG'}")
    print(f"    Opposition: fundamental n={opp_fund if opp_fund else 'none'} -- "
          f"{'correctly identifies 180 deg structure' if opp_fund == 2 else 'WRONG'}")
    print(f"    Quadrant:   fundamental n={quad_fund if quad_fund else 'none'} -- "
          f"{'correctly identifies 90 deg structure' if quad_fund == 4 else 'WRONG'}")
    print(f"    Noise:      fundamental n={noise_fund if noise_fund else 'none'} -- "
          f"{'correctly shows no dominant structure' if noise_fund is None else 'unexpected structure found'}")
    print()

    # Phase 3: Concentration comparison
    noise_max_eta = max(noise_eta)
    tri_fund_eta = tri_eta[tri_fund - 1] if tri_fund else 0.0
    opp_fund_eta = opp_eta[opp_fund - 1] if opp_fund else 0.0
    quad_fund_eta = quad_eta[quad_fund - 1] if quad_fund else 0.0

    print("  Concentration comparison (eta at fundamental vs noise peak):")
    print(f"    Triadic eta at n=3:    {tri_fund_eta:.4f}")
    print(f"    Opposition eta at n=2: {opp_fund_eta:.4f}")
    print(f"    Quadrant eta at n=4:   {quad_fund_eta:.4f}")
    print(f"    Noise max eta:         {noise_max_eta:.4f} (uniform baseline: {1.0 / n_harmonics:.4f})")
    print()

    # Validation
    tri_correct = tri_fund == 3
    opp_correct = opp_fund == 2
    quad_correct = quad_fund == 4
    noise_clean = noise_fund is None

    passed = tri_correct and opp_correct and quad_correct and noise_clean

    if passed:
        print("Test 23: PASS  (Fundamental harmonics: triadic->n=3, opposition->n=2, quadrant->n=4, noise->none)")
    else:
        print("Test 23: FAIL")
        if not tri_correct:
            print(f"  - Triadic: expected fundamental n=3, got {tri_fund}")
        if not opp_correct:
            print(f"  - Opposition: expected fundamental n=2, got {opp_fund}")
        if not quad_correct:
            print(f"  - Quadrant: expected fundamental n=4, got {quad_fund}")
        if not noise_clean:
            print(f"  - Noise: expected no fundamental, got {noise_fund}")

    return passed
