"""Tests for support of more than one contrast curve in the analysis.

When multiple contrast curves are supplied (optionally in different
photometric filters), a simulated companion is ruled out if *any* single
curve rules it out. In practice this means the limiting angular
separation adopted for each companion is the smallest (tightest) value
across all curves.

These tests exercise the pieces that combine the curves
(``triceratops.funcs``) plus the refactored priors
(``triceratops.priors``). They import the source directly so they run
without a full package install.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from triceratops.funcs import (  # noqa: E402
    file_to_contrast_curve,
    limiting_separation,
    parse_contrast_curves,
    separation_at_contrast,
)
from triceratops.priors import lnprior_background  # noqa: E402


@pytest.fixture()
def contrast_curves(tmp_path):
    """Two synthetic contrast curves; the second is uniformly deeper."""
    shallow = tmp_path / "cc_shallow.txt"
    deep = tmp_path / "cc_deep.txt"
    # columns: separation (arcsec), delta_mag
    np.savetxt(
        shallow, np.c_[[0.1, 0.5, 1.0, 2.0], [1.0, 4.0, 6.0, 8.0]],
        delimiter=","
    )
    np.savetxt(
        deep, np.c_[[0.1, 0.5, 1.0, 2.0], [3.0, 6.0, 8.0, 9.0]],
        delimiter=","
    )
    return str(shallow), str(deep)


class TestParseContrastCurves:
    def test_none_returns_none(self):
        assert parse_contrast_curves(None) == (None, None)

    def test_single_string_is_wrapped(self):
        assert parse_contrast_curves("cc.txt") == (["cc.txt"], ["TESS"])
        assert parse_contrast_curves("cc.txt", "K") == (["cc.txt"], ["K"])

    def test_single_filter_broadcasts_over_files(self):
        files, filts = parse_contrast_curves(["a.txt", "b.txt"], "H")
        assert files == ["a.txt", "b.txt"]
        assert filts == ["H", "H"]

    def test_matching_lists_are_passed_through(self):
        files, filts = parse_contrast_curves(
            ["a.txt", "b.txt"], ["J", "K"]
        )
        assert files == ["a.txt", "b.txt"]
        assert filts == ["J", "K"]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            parse_contrast_curves(["a.txt", "b.txt"], ["J", "K", "H"])


class TestLimitingSeparation:
    def test_single_curve_matches_legacy_path(self, contrast_curves):
        shallow, _ = contrast_curves
        delta_mags = np.array([2.0, 5.0, 7.0])
        legacy = separation_at_contrast(
            delta_mags, *file_to_contrast_curve(shallow)
        )
        combined = limiting_separation([delta_mags], [shallow])
        assert np.allclose(combined, legacy)

    def test_combination_takes_elementwise_minimum(self, contrast_curves):
        shallow, deep = contrast_curves
        delta_mags = np.array([2.0, 5.0, 7.0])
        s_shallow = separation_at_contrast(
            delta_mags, *file_to_contrast_curve(shallow)
        )
        s_deep = separation_at_contrast(
            delta_mags, *file_to_contrast_curve(deep)
        )
        combined = limiting_separation(
            [delta_mags, delta_mags], [shallow, deep]
        )
        assert np.allclose(combined, np.minimum(s_shallow, s_deep))
        # the deeper curve is the more constraining one here
        assert np.allclose(combined, s_deep)

    def test_different_delta_mags_per_curve(self, contrast_curves):
        """Each curve may be evaluated in its own filter, i.e. with a
        different delta_mag array."""
        shallow, deep = contrast_curves
        dm_a = np.array([3.0, 3.0, 3.0])
        dm_b = np.array([9.0, 9.0, 9.0])
        s_a = separation_at_contrast(
            dm_a, *file_to_contrast_curve(shallow)
        )
        s_b = separation_at_contrast(
            dm_b, *file_to_contrast_curve(deep)
        )
        combined = limiting_separation([dm_a, dm_b], [shallow, deep])
        assert np.allclose(combined, np.minimum(s_a, s_b))


class TestPriorsUseCombinedConstraint:
    def test_background_prior_never_looser_than_single_curve(
        self, contrast_curves
    ):
        shallow, deep = contrast_curves
        delta_mags = np.array([2.0, 5.0, 7.0])
        seps_shallow = separation_at_contrast(
            delta_mags, *file_to_contrast_curve(shallow)
        )
        seps_combined = limiting_separation(
            [delta_mags, delta_mags], [shallow, deep]
        )
        lnp_shallow = lnprior_background(100, seps_shallow)
        lnp_combined = lnprior_background(100, seps_combined)
        # tighter separation -> smaller (or equal) background probability
        assert np.all(lnp_combined <= lnp_shallow + 1e-9)
