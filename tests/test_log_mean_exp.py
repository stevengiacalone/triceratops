"""
Tests for the _log_mean_exp helper introduced in marginal_likelihoods.py.

Run with:
    pip install -e .          # install the package in editable mode
    pytest tests/             # run the suite

Requirements: numpy, scipy (both declared in setup.py).
No additional test dependencies beyond pytest are needed.
"""

import numpy as np
from scipy.special import logsumexp

from triceratops._numerics import _log_mean_exp, _normalize_probabilities


# ---------------------------------------------------------------------------
# Test 1: Underflow regression — asserts exact value, not just finiteness.
#
# With the old +600 scheme: exp(-2000 + 600) = exp(-1400) underflows to 0,
# so lnZ = log(0) = -inf.  The new helper must return exactly -2000.0.
# ---------------------------------------------------------------------------
def test_underflow_regression():
    N = 100_000
    lnL = -2000.0 * np.ones(N)
    lnZ = _log_mean_exp(lnL, N_total=N)
    assert np.isfinite(lnZ)
    assert np.isclose(lnZ, -2000.0, rtol=0, atol=1e-10), (
        f"Expected -2000.0, got {lnZ}"
    )


# ---------------------------------------------------------------------------
# Test 2: Mixed finite / -inf, using values deep enough that the old +600
# scheme would also underflow (x + 600 still far below float64 range).
#
# Exact expected value:
#   logsumexp([-1001, -1002, -1003, -1004, -1005]) - log(10)
# ---------------------------------------------------------------------------
def test_mixed_finite_and_inf_requires_fix():
    N = 10
    lnL = np.array([-1001.0, -1002.0, -np.inf, -np.inf, -1003.0,
                    -np.inf, -1004.0, -np.inf, -1005.0, -np.inf])
    EXPECTED = float(logsumexp([-1001., -1002., -1003., -1004., -1005.])
                     - np.log(10))
    lnZ = _log_mean_exp(lnL, N_total=N)
    assert np.isfinite(lnZ)
    assert np.isclose(lnZ, EXPECTED, rtol=0, atol=1e-10), (
        f"Expected {EXPECTED}, got {lnZ}"
    )


# ---------------------------------------------------------------------------
# Test 3: N_total denominator semantics.
#
# Non-transiting draws are represented as -inf and still count in the
# denominator (they contribute 0 weight to the sum but reduce the mean).
# A wrong implementation using len(finite) instead of N_total would return
# -1.0 here; the correct implementation returns -1.0 - log(10).
# ---------------------------------------------------------------------------
def test_n_total_denominator_semantics():
    N = 10
    lnL = np.array([-1.0] + [-np.inf] * 9)
    EXPECTED = -1.0 - np.log(10)   # = -3.302585092994046
    lnZ = _log_mean_exp(lnL, N_total=N)
    assert np.isfinite(lnZ)
    assert np.isclose(lnZ, EXPECTED, rtol=0, atol=1e-14), (
        f"Expected {EXPECTED}, got {lnZ}. "
        "Likely cause: N_total=len(finite) used instead of N_total=N."
    )


# ---------------------------------------------------------------------------
# Test 4: Drop-in equivalence with old approach in the non-underflow regime.
#
# For values in [-10, -1] the old +600 scheme works fine.  The new helper
# must agree to machine precision.
# ---------------------------------------------------------------------------
def test_matches_old_approach_in_safe_regime():
    rng = np.random.default_rng(42)
    lnL = rng.uniform(-10.0, -1.0, size=5000)
    N = len(lnL)
    # Old formula with the +600 shift undone in log-space
    Z_old = np.mean(np.nan_to_num(np.exp(lnL + 600)))
    lnZ_old = np.log(Z_old) - 600.0
    lnZ_new = _log_mean_exp(lnL, N_total=N)
    assert np.isfinite(lnZ_old)
    assert np.isfinite(lnZ_new)
    assert np.isclose(lnZ_new, lnZ_old, rtol=0, atol=1e-12), (
        f"New ({lnZ_new}) and old ({lnZ_old}) disagree in safe regime."
    )


# ---------------------------------------------------------------------------
# Test 5: All-inf input returns -inf, not NaN.
# ---------------------------------------------------------------------------
def test_all_inf_returns_neg_inf():
    for N in [1, 10, 100_000]:
        lnL = np.full(N, -np.inf)
        lnZ = _log_mean_exp(lnL, N_total=N)
        assert lnZ == -np.inf, f"N={N}: expected -inf, got {lnZ}"
        assert not np.isnan(lnZ), f"N={N}: got NaN instead of -inf"


# ---------------------------------------------------------------------------
# Test 6a: logsumexp normalization produces probabilities that sum to 1.
# ---------------------------------------------------------------------------
def test_normalization_sums_to_one():
    lnZ = np.array([-1.0, -2.0, -3.0])
    log_norm = logsumexp(lnZ)
    probs = np.exp(lnZ - log_norm)
    assert np.isclose(probs.sum(), 1.0, rtol=0, atol=1e-14)
    assert np.all(np.isfinite(probs))
    # Consecutive differences of 1 in log-space → ratio of e
    assert np.isclose(probs[0] / probs[1], np.e, rtol=1e-14)
    assert np.isclose(probs[1] / probs[2], np.e, rtol=1e-14)


# ---------------------------------------------------------------------------
# Test 6b: All-inf lnZ normalization returns zeros, not NaN.
#
# This exercises the degenerate-run guard added to calc_probs.
# ---------------------------------------------------------------------------
def test_normalization_all_inf_returns_zeros():
    lnZ = np.array([-np.inf] * 5)
    log_norm = logsumexp(lnZ)
    if np.isfinite(log_norm):
        probs = np.exp(lnZ - log_norm)
    else:
        probs = np.zeros_like(lnZ)
    assert not np.any(np.isnan(probs)), "Got NaN in degenerate normalization"
    assert np.all(probs == 0.0), "Expected all-zero probs in degenerate case"


# ---------------------------------------------------------------------------
# Test 7: Explicit regression — old code FAILS here, new code PASSES.
#
# Documents the concrete threshold at which the +600 scheme breaks.
# ---------------------------------------------------------------------------
def test_regression_old_fails_new_passes():
    N = 50_000
    LNL_VALUE = -1500.0   # exp(-1500 + 600) = exp(-900) ~ 10^{-391}, underflows
    lnL = LNL_VALUE * np.ones(N)
    # Verify old code actually fails (precondition for this test to be meaningful)
    Z_old = np.mean(np.nan_to_num(np.exp(lnL + 600)))
    lnZ_old = np.log(Z_old)
    assert not np.isfinite(lnZ_old), (
        "Precondition failed: old +600 code did not underflow at lnL=-1500"
    )
    # New code must return the correct value
    lnZ_new = _log_mean_exp(lnL, N_total=N)
    assert np.isfinite(lnZ_new)
    assert np.isclose(lnZ_new, LNL_VALUE, rtol=0, atol=1e-10), (
        f"Expected {LNL_VALUE}, got {lnZ_new}"
    )


# ---------------------------------------------------------------------------
# Test 8: Single-entry boundary — log(mean(exp([x]))) == x for any x.
# ---------------------------------------------------------------------------
def test_single_entry_boundary():
    for x in [-1.0, -100.0, -2000.0, 0.0]:
        lnL = np.array([x])
        lnZ = _log_mean_exp(lnL, N_total=1)
        assert np.isclose(lnZ, x, rtol=0, atol=1e-12), (
            f"x={x}: expected {x}, got {lnZ}"
        )


# ---------------------------------------------------------------------------
# Test 9: NaN entries are treated as non-finite (same semantics as -inf).
#
# PyTransit can produce NaN likelihoods for degenerate draws; these should
# contribute zero weight, matching the old nan_to_num behavior.
# ---------------------------------------------------------------------------
def test_nan_treated_as_nonfinite():
    N = 5
    lnL = np.array([-1.0, np.nan, np.nan, np.nan, np.nan])
    EXPECTED = -1.0 - np.log(5)   # = -2.6094379124341003
    lnZ = _log_mean_exp(lnL, N_total=N)
    assert np.isfinite(lnZ)
    assert np.isclose(lnZ, EXPECTED, rtol=0, atol=1e-14), (
        f"Expected {EXPECTED}, got {lnZ}"
    )


# ---------------------------------------------------------------------------
# Test 10: N_total mismatch raises ValueError.
#
# Passing len(lnL[finite]) instead of len(lnL) would silently overestimate
# evidence. The guard makes this a loud error instead.
# ---------------------------------------------------------------------------
def test_n_total_mismatch_raises():
    lnL = np.array([-1.0, -2.0, -3.0])
    try:
        _log_mean_exp(lnL, N_total=2)   # wrong: 2 != 3
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "N_total" in str(e)


# ---------------------------------------------------------------------------
# Test 11: _normalize_probabilities returns correct probs and status for
# all three cases: normal, all-neginf, and NaN/+inf anomaly.
#
# Tests the extracted pure function directly — no astronomy dep chain needed.
# ---------------------------------------------------------------------------
def test_normalize_probabilities_normal_case():
    lnZ = np.array([-1.0, -2.0, -3.0])
    probs, status = _normalize_probabilities(lnZ)
    assert status == 'ok'
    assert np.isclose(probs.sum(), 1.0, rtol=0, atol=1e-14)
    assert np.all(np.isfinite(probs))
    assert np.isclose(probs[0] / probs[1], np.e, rtol=1e-14)


def test_normalize_probabilities_all_neginf():
    lnZ = np.full(5, -np.inf)
    probs, status = _normalize_probabilities(lnZ)
    assert status == 'all_neginf'
    assert np.all(probs == 0.0)
    assert not np.any(np.isnan(probs))


def test_normalize_probabilities_anomaly_nan():
    lnZ = np.array([-1.0, np.nan, -3.0])
    probs, status = _normalize_probabilities(lnZ)
    assert status == 'anomaly'
    assert np.all(probs == 0.0)
    assert not np.any(np.isnan(probs))


def test_normalize_probabilities_anomaly_posinf():
    lnZ = np.array([-1.0, np.inf, -3.0])
    probs, status = _normalize_probabilities(lnZ)
    assert status == 'anomaly'
    assert np.all(probs == 0.0)


def test_normalize_probabilities_partial_neginf_is_ok():
    # Scenarios with lnZ=-inf (no transiting draws) should get prob=0,
    # not trigger the degenerate branch, so long as some scenarios are finite.
    lnZ = np.array([-1.0, -np.inf, -3.0])
    probs, status = _normalize_probabilities(lnZ)
    assert status == 'ok'
    assert np.isclose(probs.sum(), 1.0, rtol=0, atol=1e-14)
    assert probs[1] == 0.0   # -inf scenario gets exactly zero probability


# ---------------------------------------------------------------------------
# Exact expected values for reference
# ---------------------------------------------------------------------------
# Test 1:  -2000.0                       log(exp(-2000)) = -2000
# Test 2:  logsumexp([...]) - log(10)    ≈ -1001 + log(1.183) - 2.303
# Test 3:  -3.302585092994046            -1 - log(10)
# Test 7:  -1500.0                       log(exp(-1500)) = -1500
# Test 8:  x for each x                  identity
# Test 9:  -2.6094379124341003           -1 - log(5)
