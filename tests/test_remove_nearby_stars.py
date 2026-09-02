"""Tests for target.remove_nearby_stars().

remove_nearby_stars() drops every star except the target from the
.stars dataframe, for use when follow-up observations have shown the
transit to be on-target. It should keep the target's (diluted) flux
ratio and keep the per-sector pixel-coordinate arrays consistent.

The full triceratops.triceratops import is heavy (lightkurve, astroquery,
pytransit), so this module is marked ``heavy``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.heavy


def _fake_target(with_depths: bool):
    """A target instance with a hand-built four-star field (index 0 is
    the target), bypassing the network queries in __init__."""
    from triceratops.triceratops import target

    tgt = target.__new__(target)
    cols = dict(
        ID=["100", "200", "300", "400"],
        Tmag=[10.0, 15.0, 16.0, 17.0],
        mass=[1.0, 0.5, 0.5, 0.5],
        rad=[1.0, 0.5, 0.5, 0.5],
        Teff=[5800.0, 3800.0, 3800.0, 3800.0],
        plx=[10.0, 1.0, 1.0, 1.0],
        fluxratio=[0.7, 0.2, 0.07, 0.03],
    )
    if with_depths:
        cols["tdepth"] = [0.007, 0.03, 0.0, 0.0]
    tgt.stars = pd.DataFrame(cols)
    tgt.pix_coords = [
        np.arange(8, dtype=float).reshape(4, 2),
        np.arange(100, 108, dtype=float).reshape(4, 2),
    ]
    return tgt


class TestRemoveNearbyStars:
    def test_keeps_only_the_target(self):
        tgt = _fake_target(with_depths=True)
        tgt.remove_nearby_stars()
        assert len(tgt.stars) == 1
        assert tgt.stars["ID"].values[0] == "100"

    def test_preserves_target_flux_ratio_and_depth(self):
        tgt = _fake_target(with_depths=True)
        tgt.remove_nearby_stars()
        # the diluted flux ratio must survive so calc_probs still
        # corrects the light curve for blending
        assert tgt.stars["fluxratio"].values[0] == 0.7
        assert tgt.stars["tdepth"].values[0] == 0.007

    def test_pix_coords_truncated_to_target(self):
        tgt = _fake_target(with_depths=True)
        tgt.remove_nearby_stars()
        assert [pc.shape for pc in tgt.pix_coords] == [(1, 2), (1, 2)]
        assert np.array_equal(tgt.pix_coords[0][0], np.array([0.0, 1.0]))

    def test_index_is_reset(self):
        tgt = _fake_target(with_depths=True)
        tgt.remove_nearby_stars()
        assert list(tgt.stars.index) == [0]

    def test_warns_when_run_before_calc_depths(self, capsys):
        tgt = _fake_target(with_depths=False)
        tgt.remove_nearby_stars()
        out = capsys.readouterr().out
        assert "before" in out and "calc_depths" in out
        assert len(tgt.stars) == 1

    def test_no_warning_after_calc_depths(self, capsys):
        tgt = _fake_target(with_depths=True)
        tgt.remove_nearby_stars()
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_isolated_target_is_a_no_op(self):
        tgt = _fake_target(with_depths=True)
        tgt.stars = tgt.stars.iloc[:1]
        tgt.pix_coords = [pc[:1] for pc in tgt.pix_coords]
        tgt.remove_nearby_stars()
        assert len(tgt.stars) == 1
