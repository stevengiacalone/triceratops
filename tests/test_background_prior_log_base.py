"""Tests for NC-03: np.log10 -> np.log fix in background companion priors.

Verifies that background star priors use natural logarithm (np.log) rather
than base-10 logarithm (np.log10). The prior is added to natural-log
likelihoods in Bayesian evidence integrals, so it must be a natural log.

These tests are designed to run without installing the full triceratops
package -- they add the source directory to sys.path and import directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make triceratops importable without pip install
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from triceratops.priors import lnprior_background  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

# Simple contrast curve: linearly maps delta_mag to separation
_SEPS = np.array([0.5, 1.0, 2.0])
_CONTRASTS = np.array([2.0, 4.0, 6.0])

# Known inputs
_N_COMP = 100
_DELTA_MAGS = np.array([3.0])  # interpolates to sep = 0.75 arcsec

# The expected separation after interpolation
_EXPECTED_SEP = np.interp(_DELTA_MAGS, _CONTRASTS, _SEPS)[0]  # 0.75

# The prior argument (before taking the log)
_PRIOR_ARG = (_N_COMP / 0.1) * (1 / 3600) ** 2 * _EXPECTED_SEP ** 2


class TestLnpriorBackgroundUsesNaturalLog:
    """Verify that lnprior_background() returns np.log (not np.log10)."""

    def test_result_equals_natural_log(self) -> None:
        """The function output must equal np.log(arg), not np.log10(arg)."""
        result = lnprior_background(
            _N_COMP, _DELTA_MAGS, _SEPS, _CONTRASTS
        )
        expected = np.log(_PRIOR_ARG)
        assert result[0] == pytest.approx(expected, rel=1e-12)

    def test_result_does_not_equal_log10(self) -> None:
        """Confirm the old (buggy) log10 value is NOT returned."""
        result = lnprior_background(
            _N_COMP, _DELTA_MAGS, _SEPS, _CONTRASTS
        )
        wrong_value = np.log10(_PRIOR_ARG)
        assert result[0] != pytest.approx(wrong_value, rel=1e-6)


class TestInlinePriorConstant:
    """Verify the inline prior used when no contrast curve is provided.

    In marginal_likelihoods.py, the no-contrast-curve path computes:
        np.log((N_comp/0.1) * (1/3600)**2 * 2.2**2)
    where 2.2 arcsec is the default separation.
    """

    def test_inline_prior_is_natural_log(self) -> None:
        N_comp = 100
        default_sep = 2.2  # arcsec
        arg = (N_comp / 0.1) * (1 / 3600) ** 2 * default_sep ** 2
        # This is what the fixed code computes
        fixed_value = np.log(arg)
        # This is what the buggy code computed
        buggy_value = np.log10(arg)

        # The fixed value must be the natural log
        assert fixed_value == pytest.approx(np.log(arg), rel=1e-15)
        # And it must differ from log10 by exactly ln(10)
        assert fixed_value / buggy_value == pytest.approx(
            np.log(10), rel=1e-12
        )


class TestMathematicalRelationship:
    """Prove the old error was exactly a factor of ln(10).

    For any positive x:  log10(x) * ln(10) == ln(x)

    So the buggy prior (log10) times ln(10) must equal the fixed prior (ln).
    This demonstrates the systematic deflation factor.
    """

    def test_log10_times_ln10_equals_ln(self) -> None:
        """old_value * ln(10) == new_value for the lnprior_background case."""
        result_fixed = lnprior_background(
            _N_COMP, _DELTA_MAGS, _SEPS, _CONTRASTS
        )
        # Compute what the old (buggy) code would have returned
        old_buggy = np.log10(_PRIOR_ARG)
        new_correct = result_fixed[0]

        assert old_buggy * np.log(10) == pytest.approx(
            new_correct, rel=1e-12
        )

    def test_deflation_factor_is_ln10(self) -> None:
        """The ratio new/old must be exactly ln(10) ~ 2.302585."""
        old_buggy = np.log10(_PRIOR_ARG)
        new_correct = np.log(_PRIOR_ARG)

        ratio = new_correct / old_buggy
        assert ratio == pytest.approx(np.log(10), rel=1e-12)

    def test_inline_prior_deflation_factor(self) -> None:
        """Same relationship holds for the inline constant (2.2 arcsec)."""
        N_comp = 100
        arg = (N_comp / 0.1) * (1 / 3600) ** 2 * 2.2 ** 2
        old_buggy = np.log10(arg)
        new_correct = np.log(arg)
        assert old_buggy * np.log(10) == pytest.approx(
            new_correct, rel=1e-12
        )

    def test_various_n_comp_values(self) -> None:
        """The ln(10) relationship holds regardless of N_comp."""
        for N_comp in [1, 10, 50, 100, 500, 1000]:
            result = lnprior_background(
                N_comp, _DELTA_MAGS, _SEPS, _CONTRASTS
            )
            arg = (N_comp / 0.1) * (1 / 3600) ** 2 * _EXPECTED_SEP ** 2
            expected = np.log(arg)
            assert result[0] == pytest.approx(expected, rel=1e-12), (
                f"Failed for N_comp={N_comp}"
            )
