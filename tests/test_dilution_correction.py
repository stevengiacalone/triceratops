"""Tests for the light-curve dilution handling in calc_depths()/calc_probs().

As of 1.1.0 the input light curve is assumed to be dilution-corrected
(on-target) by default; calc_depths() and calc_probs() take a
dilution_corrected flag, and each nearby star's light curve is derived
from the target's rather than by correcting the input per star.

Tests that construct a ``target`` are marked ``heavy``; the flux-transform
arithmetic tests only need ``renorm_flux``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# target flux ratio 0.8, one neighbor at 0.2 (target 4x brighter)
_DMAG = 2.5 * np.log10(4.0)


def _target_two_stars():
    """A target with two stars stacked at the centre of a 5x5 aperture,
    so calc_depths() yields fluxratios of 0.8 (target) and 0.2, plus the
    aperture to pass to calc_depths()."""
    from triceratops.triceratops import target

    tgt = target.__new__(target)
    tgt.stars = pd.DataFrame({
        "ID": ["T", "N"],
        "Tmag": [10.0, 10.0 + _DMAG],
        "mass": [1.0, 0.5],
        "rad": [1.0, 0.5],
        "Teff": [5800.0, 3800.0],
        "plx": [10.0, 1.0],
    })
    centre = 2.0
    tgt.pix_coords = [np.array([[centre, centre], [centre, centre]])]
    aperture = [
        np.array([[x, y] for x in range(5) for y in range(5)], dtype=float)
        ]
    return tgt, aperture


@pytest.mark.heavy
class TestCalcDepths:
    def test_fluxratios(self):
        tgt, ap = _target_two_stars()
        tgt.calc_depths(tdepth=0.01, all_ap_pixels=ap)
        assert tgt.stars["fluxratio"].values[0] == pytest.approx(0.8, abs=1e-6)
        assert tgt.stars["fluxratio"].values[1] == pytest.approx(0.2, abs=1e-6)

    def test_corrected_depth_is_on_target_depth(self):
        tgt, ap = _target_two_stars()
        tgt.calc_depths(tdepth=0.01, all_ap_pixels=ap,
                        dilution_corrected=True)
        # target's required intrinsic depth == the input depth
        assert tgt.stars["tdepth"].values[0] == pytest.approx(0.01, rel=1e-6)
        # neighbour: input * F_target / F_neighbour = 0.01 * 0.8/0.2
        assert tgt.stars["tdepth"].values[1] == pytest.approx(0.04, rel=1e-6)

    def test_uncorrected_depth_is_aperture_depth(self):
        tgt, ap = _target_two_stars()
        tgt.calc_depths(tdepth=0.008, all_ap_pixels=ap,
                        dilution_corrected=False)
        # raw aperture dip 0.008 -> target needs 0.008/0.8, neighbour 0.008/0.2
        assert tgt.stars["tdepth"].values[0] == pytest.approx(0.01, rel=1e-6)
        assert tgt.stars["tdepth"].values[1] == pytest.approx(0.04, rel=1e-6)

    def test_impossible_depth_zeroed(self):
        tgt, ap = _target_two_stars()
        # on-target depth 0.3 -> neighbour would need 1.2 -> impossible -> 0
        tgt.calc_depths(tdepth=0.3, all_ap_pixels=ap)
        assert tgt.stars["tdepth"].values[1] == 0.0


class TestFluxTransform:
    """The per-star light-curve transform used inside calc_probs()."""

    @staticmethod
    def _neighbour_lc(flux0, ferr0, f_target, f_i, dilution_corrected):
        from triceratops.funcs import renorm_flux

        if not dilution_corrected:
            flux0, ferr0 = renorm_flux(flux0, ferr0, f_target)
        flux = 1.0 + (f_target / f_i) * (flux0 - 1.0)
        ferr = ferr0 * f_target / f_i
        return flux, ferr

    def test_target_lc_unchanged_when_corrected(self):
        corrected = np.array([1.0, 0.99])  # 1% on-target transit
        flux, _ = self._neighbour_lc(corrected, 1e-3, 0.8, 0.8, True)
        assert flux == pytest.approx(corrected)

    def test_neighbour_depth_scales_by_flux_ratio(self):
        corrected = np.array([1.0, 0.99])
        flux, ferr = self._neighbour_lc(corrected, 1e-3, 0.8, 0.2, True)
        # 1% -> 1% * 0.8/0.2 = 4%
        assert flux == pytest.approx([1.0, 0.96])
        assert ferr == pytest.approx(1e-3 * 4.0)

    def test_raw_input_matches_corrected_input(self):
        # a raw aperture curve with the same event should give the same
        # per-star light curves once the flag is set accordingly
        f_t, f_n = 0.8, 0.2
        on_target_depth = 0.01
        corrected = np.array([1.0, 1.0 - on_target_depth])
        raw = np.array([1.0, 1.0 - on_target_depth * f_t])
        for f_i in (f_t, f_n):
            a, _ = self._neighbour_lc(corrected, 1e-3, f_t, f_i, True)
            b, _ = self._neighbour_lc(raw, 1e-3, f_t, f_i, False)
            assert a == pytest.approx(b)
