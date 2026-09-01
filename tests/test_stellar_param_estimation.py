"""Tests for estimating missing stellar parameters from photometry.

triceratops.funcs.estimate_stellar_parameters() fills in a star's mass,
radius, and/or Teff from broadband photometry (Gaia, 2MASS, Johnson V)
using the main-sequence (dwarf) sequence of Pecaut & Mamajek (2013),
with a Stefan-Boltzmann radius when a parallax is available.

These tests use fixed photometry (no network) so they run offline. The
reference values are the TIC v8.2 parameters for the corresponding
stars.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.funcs import estimate_stellar_parameters


# TIC 307210830 -- a nearby mid-M dwarf.
# TIC values: mass 0.293 Msun, rad 0.314 Rsun, Teff 3429 K.
M_DWARF = dict(
    Vmag=11.685, Gmag=10.5976, BPmag=11.9775, RPmag=9.47197,
    Jmag=7.933, Hmag=7.359, Kmag=7.101, plx=94.1385, ebv=0.0,
)
M_DWARF_TRUTH = dict(mass=0.293, rad=0.314, Teff=3429.0)


class TestFullPhotometry:
    def test_recovers_m_dwarf_parameters(self):
        res = estimate_stellar_parameters(**M_DWARF)
        assert set(res["estimated"]) == {"mass", "rad", "Teff"}
        assert res["Teff"] == pytest.approx(M_DWARF_TRUTH["Teff"], abs=200)
        assert res["mass"] == pytest.approx(M_DWARF_TRUTH["mass"], rel=0.15)
        assert res["rad"] == pytest.approx(M_DWARF_TRUTH["rad"], rel=0.15)

    def test_uses_parallax_for_radius(self):
        res = estimate_stellar_parameters(**M_DWARF)
        assert "Stefan-Boltzmann" in res["method"]["rad"]

    def test_prefers_gaia_color_for_teff(self):
        res = estimate_stellar_parameters(**M_DWARF)
        assert res["method"]["Teff"] == "BP-RP color"


class TestPartialInputs:
    def test_known_values_are_preserved(self):
        res = estimate_stellar_parameters(
            mass=0.5, rad=0.5, Teff=3800.0, **M_DWARF
        )
        assert res["estimated"] == []
        assert (res["mass"], res["rad"], res["Teff"]) == (0.5, 0.5, 3800.0)

    def test_known_teff_used_as_anchor(self):
        # only Teff known, no photometry at all -> mass & rad from the
        # dwarf sequence at that Teff
        res = estimate_stellar_parameters(Teff=5772.0)
        assert set(res["estimated"]) == {"mass", "rad"}
        assert res["mass"] == pytest.approx(1.0, rel=0.1)
        assert res["rad"] == pytest.approx(1.0, rel=0.1)

    def test_only_missing_params_are_filled(self):
        res = estimate_stellar_parameters(rad=0.9, **M_DWARF)
        assert "rad" not in res["estimated"]
        assert res["rad"] == 0.9
        assert set(res["estimated"]) == {"mass", "Teff"}

    def test_radius_falls_back_to_sequence_without_parallax(self):
        phot = {k: v for k, v in M_DWARF.items() if k != "plx"}
        res = estimate_stellar_parameters(**phot)
        assert res["method"]["rad"] == "dwarf sequence (Teff)"
        assert np.isfinite(res["rad"])


class TestInsufficientData:
    def test_no_usable_photometry_returns_nan(self):
        res = estimate_stellar_parameters(Kmag=10.0)
        assert res["estimated"] == []
        assert np.isnan(res["mass"])
        assert np.isnan(res["rad"])
        assert np.isnan(res["Teff"])

    def test_handles_none_and_strings_gracefully(self):
        res = estimate_stellar_parameters(
            BPmag=None, RPmag="", Jmag=7.933, Hmag=7.359, Kmag=7.101,
            plx=94.1385,
        )
        assert np.isfinite(res["Teff"])
        assert np.isfinite(res["mass"])


class TestEvolvedStars:
    def test_low_logg_flags_evolved_and_assigns_solar_mass(self):
        res = estimate_stellar_parameters(
            Teff=4700.0, logg=2.5, Vmag=8.0, Kmag=5.9, plx=3.33
        )
        assert res["evolved"] is True
        assert res["mass"] == 1.0
        assert res["method"]["mass"] == "assumed (evolved star)"

    def test_radius_far_above_main_sequence_flags_evolved(self):
        # a star with a known radius ~3.7x the main-sequence value at
        # its Teff (a la TOI-197): only the mass is missing
        res = estimate_stellar_parameters(
            rad=2.90, Teff=5083.0, Vmag=8.15, Kmag=6.6, plx=10.4
        )
        assert res["evolved"] is True
        assert res["estimated"] == ["mass"]
        assert res["mass"] == 1.0
        assert res["rad"] == 2.90  # provided radius is kept

    def test_evolved_radius_needs_parallax(self):
        # evolved (low logg) but no parallax -> radius left unestimated
        # rather than taking a wildly wrong dwarf-sequence value
        res = estimate_stellar_parameters(Teff=4700.0, logg=2.2)
        assert res["evolved"] is True
        assert np.isnan(res["rad"])
        assert res["mass"] == 1.0

    def test_evolved_with_parallax_gets_stefan_boltzmann_radius(self):
        res = estimate_stellar_parameters(
            Teff=4700.0, logg=2.5, Kmag=5.96, plx=3.33
        )
        assert res["evolved"] is True
        assert "Stefan-Boltzmann" in res["method"]["rad"]
        assert res["rad"] > 5.0  # unmistakably a giant

    def test_main_sequence_star_not_flagged(self):
        res = estimate_stellar_parameters(**M_DWARF)
        assert res["evolved"] is False


class TestSunLikeStar:
    def test_solar_analog(self):
        # Sun at 10 pc: V ~ 4.83 + 5 = ... use apparent mags consistent
        # with M_V(G2V)=4.80, M_Ks=3.236 and a 10 mas parallax (100 pc,
        # distance modulus 5.0)
        res = estimate_stellar_parameters(
            Vmag=4.80 + 5.0, Kmag=3.236 + 5.0, plx=10.0, ebv=0.0
        )
        assert res["Teff"] == pytest.approx(5770, abs=250)
        assert res["mass"] == pytest.approx(1.0, rel=0.1)
        assert res["rad"] == pytest.approx(1.0, rel=0.12)
