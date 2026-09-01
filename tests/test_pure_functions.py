"""Unit tests for the pure helper functions in triceratops.funcs and
triceratops.priors.

These exercise the small, deterministic building blocks (flux
renormalization, the mass-radius-Teff main-sequence relations, the
photometric colour relation) and sanity-check the prior samplers and
log-prior functions (ranges, shapes, monotonicity). None of this needs
the full triceratops install.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.funcs import (
    color_Teff_relations,
    flux_relation,
    renorm_flux,
    stellar_relations,
)
from triceratops.priors import (
    lnprior_Mstar_binary,
    lnprior_Mstar_planet,
    lnprior_Porb_binary,
    lnprior_Porb_planet,
    lnprior_bound_EB,
    lnprior_bound_TP,
    sample_ecc,
    sample_inc,
    sample_q,
    sample_q_companion,
    sample_rp,
    sample_w,
)


# ---------------------------------------------------------------------------
# funcs.renorm_flux
# ---------------------------------------------------------------------------

class TestRenormFlux:
    def test_out_of_transit_normalized_to_one(self):
        flux = np.array([1.0, 1.0, 1.0])
        out, _ = renorm_flux(flux, 0.001, 0.5)
        assert out == pytest.approx(1.0)

    def test_transit_depth_is_deblended(self):
        # aperture dip of 1% with the star contributing 25% of the flux
        # -> the isolated star would show a 4% dip
        flux = np.array([1.0, 0.99])
        out, _ = renorm_flux(flux, 0.001, 0.25)
        assert out == pytest.approx([1.0, 0.96])

    def test_error_scales_inversely_with_flux_ratio(self):
        _, err = renorm_flux(np.array([1.0]), 0.002, 0.5)
        assert err == pytest.approx(0.004)

    def test_identity_when_star_is_all_the_flux(self):
        flux = np.array([1.0, 0.987, 1.0])
        out, err = renorm_flux(flux, 0.003, 1.0)
        assert out == pytest.approx(flux)
        assert err == pytest.approx(0.003)


# ---------------------------------------------------------------------------
# funcs.stellar_relations
# ---------------------------------------------------------------------------

class TestStellarRelations:
    def test_solar_mass_gives_roughly_solar_star(self):
        big = np.full(1, 1e9)
        radii, teffs = stellar_relations(np.array([1.0]), big, big)
        assert radii[0] == pytest.approx(1.0, abs=0.2)
        assert teffs[0] == pytest.approx(5772, abs=400)

    def test_monotonic_in_mass(self):
        masses = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
        big = np.full(masses.size, 1e9)
        radii, teffs = stellar_relations(masses, big, big)
        assert np.all(np.diff(radii) > 0)
        assert np.all(np.diff(teffs) > 0)

    def test_floor_values_applied(self):
        radii, teffs = stellar_relations(
            np.array([0.05]), np.full(1, 1e9), np.full(1, 1e9)
        )
        assert radii[0] >= 0.1
        assert teffs[0] >= 2800

    def test_capped_by_max_arguments(self):
        radii, teffs = stellar_relations(
            np.array([1.5]), np.array([0.5]), np.array([4000.0])
        )
        assert radii[0] <= 0.5
        assert teffs[0] <= 4000.0


# ---------------------------------------------------------------------------
# funcs.flux_relation
# ---------------------------------------------------------------------------

class TestFluxRelation:
    def test_reference_mass_gives_unity(self):
        # the TESS-band spline is anchored so 0.9 Msun -> flux ratio 1
        assert flux_relation(np.array([0.9]))[0] == pytest.approx(1.0)

    def test_monotonic_increasing_with_mass(self):
        masses = np.array([0.2, 0.5, 1.0, 2.0])
        for filt in ("TESS", "J", "H", "K"):
            fluxes = flux_relation(masses, filt)
            assert np.all(np.diff(fluxes) > 0), filt

    def test_low_mass_star_is_faint(self):
        assert flux_relation(np.array([0.2]))[0] < 0.1

    def test_filters_differ_for_cool_stars(self):
        m = np.array([0.3])
        assert flux_relation(m, "K")[0] != flux_relation(m, "TESS")[0]


# ---------------------------------------------------------------------------
# funcs.color_Teff_relations
# ---------------------------------------------------------------------------

class TestColorTeffRelations:
    def test_blue_star_is_hot(self):
        teff = color_Teff_relations(10.0, 8.5)  # V-Ks = 1.5
        assert 5500 < teff < 8000

    def test_red_star_is_cool(self):
        teff = color_Teff_relations(15.0, 8.0)  # V-Ks = 7.0, > 5.05 branch
        assert 2500 < teff < 3600

    def test_cooler_color_gives_lower_teff(self):
        assert (color_Teff_relations(10.0, 6.0)
                > color_Teff_relations(10.0, 4.0))


# ---------------------------------------------------------------------------
# priors – samplers
# ---------------------------------------------------------------------------

class TestSamplers:
    def test_sample_w_spans_full_circle(self):
        w = sample_w(np.array([0.0, 0.25, 0.5, 1.0]))
        assert w[0] == 0.0
        assert w[-1] == pytest.approx(360.0)

    def test_sample_inc_within_bounds_and_monotonic(self):
        x = np.linspace(0.0, 1.0, 50)
        inc = sample_inc(x.copy())
        assert np.all((inc >= 0) & (inc <= 90))
        assert np.all(np.diff(inc) > 0)

    def test_sample_inc_favours_edge_on(self):
        rng = np.random.default_rng(0)
        inc = sample_inc(rng.random(20000))
        # sin(i) weighting -> more than half the draws above 60 deg
        assert np.mean(inc > 60) > 0.5

    def test_sample_ecc_planet_in_unit_interval(self):
        e = sample_ecc(np.zeros(500), planet=True, P_orb=5.0)
        assert e.shape == (500,)
        assert np.all((e >= 0) & (e < 1))

    def test_sample_ecc_binary_period_dependent(self):
        short = sample_ecc(np.zeros(2000), planet=False, P_orb=3.0)
        long = sample_ecc(np.zeros(2000), planet=False, P_orb=50.0)
        assert np.all((short >= 0) & (short < 1))
        assert np.all((long >= 0) & (long < 1))
        # the long-period power law is less bottom-heavy
        assert np.mean(long) > np.mean(short)

    def test_sample_rp_within_configured_range(self):
        x = np.linspace(1e-4, 1 - 1e-4, 400)
        rp = sample_rp(x.copy(), np.full(x.size, 1.0), flatpriors=False)
        assert np.all((rp >= 0.5) & (rp <= 20.0))
        rp_flat = sample_rp(x.copy(), np.full(x.size, 1.0), flatpriors=True)
        assert np.all((rp_flat >= 0.5) & (rp_flat <= 20.0))

    def test_sample_q_returns_valid_mass_ratios(self):
        x = np.linspace(1e-4, 1 - 1e-4, 400)
        for fn in (sample_q, sample_q_companion):
            q = fn(x.copy(), 1.0)
            assert q.shape == x.shape
            assert np.all((q > 0) & (q <= 1.0 + 1e-9))


# ---------------------------------------------------------------------------
# priors – log-prior functions
# ---------------------------------------------------------------------------

class TestLogPriors:
    def test_lnprior_mstar_planet_is_zeroed(self):
        # currently returns 0.0 by design (occurrence-rate term omitted)
        assert lnprior_Mstar_planet(np.array([0.3, 1.0, 2.0])) == 0.0

    def test_lnprior_mstar_binary_finite(self):
        val = lnprior_Mstar_binary(np.array([0.5, 1.0, 1.5]))
        assert np.all(np.isfinite(val))

    def test_lnprior_porb_finite_for_reasonable_periods(self):
        assert np.isfinite(lnprior_Porb_planet(5.0, flatpriors=False))
        assert np.isfinite(lnprior_Porb_planet(5.0, flatpriors=True))
        assert np.isfinite(lnprior_Porb_binary(5.0))

    def test_lnprior_bound_increases_with_separation(self):
        tight = np.full(3, 0.05)
        wide = np.full(3, 3.0)
        for fn in (lnprior_bound_TP, lnprior_bound_EB):
            assert np.all(fn(1.0, 10.0, wide) >= fn(1.0, 10.0, tight) - 1e-9)

    def test_lnprior_bound_handles_nan_parallax(self):
        out = lnprior_bound_TP(1.0, np.nan, np.full(2, 1.0))
        assert np.all(np.isfinite(out) | (out == -np.inf))
