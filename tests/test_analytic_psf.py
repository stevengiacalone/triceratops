"""
Tests for the analytic PSF integral (NC-02: dblquad -> ndtr).

Verifies that replacing scipy.integrate.dblquad(Gauss2D, ...) with the
closed-form ndtr computation produces identical results and correct behaviour.

The 2D Gaussian PSF integral over a pixel box [x0, x1] x [y0, y1] is:

    A * [Phi((x1 - mu_x)/sigma) - Phi((x0 - mu_x)/sigma)]
      * [Phi((y1 - mu_y)/sigma) - Phi((y0 - mu_y)/sigma)]

where Phi = scipy.special.ndtr (standard normal CDF).  This is the exact
analytic solution, not an approximation.

Run with:
    pytest tests/test_analytic_psf.py -v
"""

import inspect
import os
import sys

import numpy as np
from scipy.integrate import dblquad
from scipy.special import ndtr


# ---- helpers ----

def gauss2d(x, y, mu_x, mu_y, sigma, A):
    """Reproduce the original Gauss2D from funcs.py for reference testing."""
    exponent = ((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
    return A / (2 * np.pi * sigma**2) * np.exp(-exponent)


def flux_dblquad(pixels, mu_x, mu_y, sigma, A):
    """Compute flux via dblquad (the old method)."""
    total = 0.0
    for px, py in pixels:
        val, _ = dblquad(
            gauss2d,
            py - 0.5, py + 0.5,
            px - 0.5, px + 0.5,
            args=(mu_x, mu_y, sigma, A),
        )
        total += val
    return total


def flux_ndtr(pixels, mu_x, mu_y, sigma, A):
    """Compute flux via ndtr (the new method)."""
    pixels = np.asarray(pixels)
    return A * np.sum(
        (ndtr((pixels[:, 0] + 0.5 - mu_x) / sigma)
         - ndtr((pixels[:, 0] - 0.5 - mu_x) / sigma))
        * (ndtr((pixels[:, 1] + 0.5 - mu_y) / sigma)
         - ndtr((pixels[:, 1] - 0.5 - mu_y) / sigma))
    )


# ---- test 1: equivalence with dblquad ----

def test_equivalence_with_dblquad():
    """ndtr and dblquad give identical results within dblquad's tolerance."""
    mu_x, mu_y = 5.3, 4.7
    sigma = 0.75
    A = 1.0

    # 5x5 pixel grid centered on (5, 5)
    pixels = np.array([
        [x, y]
        for x in range(3, 8)
        for y in range(2, 7)
    ])

    val_dblquad = flux_dblquad(pixels, mu_x, mu_y, sigma, A)
    val_ndtr = flux_ndtr(pixels, mu_x, mu_y, sigma, A)

    assert np.isclose(val_ndtr, val_dblquad, rtol=0, atol=1e-8), (
        f"ndtr={val_ndtr}, dblquad={val_dblquad}, "
        f"diff={abs(val_ndtr - val_dblquad)}"
    )


# ---- test 2: centered star captures nearly all flux ----

def test_centered_star_known_value():
    """A star centered on pixel (5,5) with a 3x3 aperture captures most flux."""
    mu_x, mu_y = 5.0, 5.0
    sigma = 0.75
    A = 1.0

    # 3x3 grid centered on (5, 5)
    pixels = np.array([
        [x, y]
        for x in range(4, 7)
        for y in range(4, 7)
    ])

    val = flux_ndtr(pixels, mu_x, mu_y, sigma, A)

    # Analytic expected value: the integral of a 2D Gaussian over [-1.5, 1.5]^2
    # in units of sigma: 1.5 / 0.75 = 2.0
    # Phi(2) - Phi(-2) = 0.97725 - 0.02275 = 0.95450
    # For each axis, so total = 0.95450^2 = 0.91107
    expected = (ndtr(2.0) - ndtr(-2.0))**2
    assert np.isclose(val, expected, rtol=0, atol=1e-12), (
        f"val={val}, expected={expected}"
    )
    # Sanity: should be close to 1 but not exactly 1
    assert 0.90 < val < 1.0, f"val={val} out of expected range"


# ---- test 3: off-center star has less captured flux ----

def test_off_center_star():
    """A star offset from the aperture center captures less flux."""
    sigma = 0.75
    A = 1.0

    # Same 3x3 aperture
    pixels = np.array([
        [x, y]
        for x in range(4, 7)
        for y in range(4, 7)
    ])

    flux_centered = flux_ndtr(pixels, 5.0, 5.0, sigma, A)
    flux_offset = flux_ndtr(pixels, 5.3, 4.7, sigma, A)

    assert flux_offset > 0, "Off-center flux should be positive"
    assert flux_offset < flux_centered, (
        f"Off-center flux ({flux_offset}) should be less than "
        f"centered flux ({flux_centered})"
    )


# ---- test 4: multiple stars, flux ratios sum to 1 ----

def test_multiple_stars_flux_ratios():
    """Two stars in the same aperture: flux ratios sum to 1 after normalization."""
    sigma = 0.75

    # 5x5 aperture centered on (5, 5)
    pixels = np.array([
        [x, y]
        for x in range(3, 8)
        for y in range(3, 8)
    ])

    # Star 1: bright (A=1.0), centered
    flux_1 = flux_ndtr(pixels, 5.0, 5.0, sigma, 1.0)
    # Star 2: dimmer (A=0.1), slightly offset
    flux_2 = flux_ndtr(pixels, 6.0, 5.5, sigma, 0.1)

    total = flux_1 + flux_2
    ratio_1 = flux_1 / total
    ratio_2 = flux_2 / total

    assert np.isclose(ratio_1 + ratio_2, 1.0, rtol=0, atol=1e-14), (
        f"Ratios sum to {ratio_1 + ratio_2}, expected 1.0"
    )
    # The brighter star should dominate
    assert ratio_1 > ratio_2, (
        f"Bright star ratio ({ratio_1}) should exceed dim star ({ratio_2})"
    )


# ---- test 5: equivalence across a range of star positions ----

def test_equivalence_sweep():
    """Sweep star positions and verify ndtr matches dblquad everywhere."""
    sigma = 0.75
    A = 0.6

    # 3x3 pixel grid
    pixels = np.array([
        [x, y]
        for x in range(4, 7)
        for y in range(4, 7)
    ])

    rng = np.random.default_rng(12345)
    for _ in range(20):
        mu_x = rng.uniform(3.0, 7.0)
        mu_y = rng.uniform(3.0, 7.0)

        val_dblquad = flux_dblquad(pixels, mu_x, mu_y, sigma, A)
        val_ndtr = flux_ndtr(pixels, mu_x, mu_y, sigma, A)

        assert np.isclose(val_ndtr, val_dblquad, rtol=0, atol=1e-8), (
            f"mu=({mu_x:.4f}, {mu_y:.4f}): "
            f"ndtr={val_ndtr}, dblquad={val_dblquad}, "
            f"diff={abs(val_ndtr - val_dblquad)}"
        )


# ---- test 6: single pixel integral ----

def test_single_pixel():
    """ndtr integral over a single pixel matches dblquad exactly."""
    mu_x, mu_y = 5.2, 5.8
    sigma = 0.75
    A = 1.0

    pixels = np.array([[5, 6]])

    val_dblquad = flux_dblquad(pixels, mu_x, mu_y, sigma, A)
    val_ndtr = flux_ndtr(pixels, mu_x, mu_y, sigma, A)

    assert np.isclose(val_ndtr, val_dblquad, rtol=0, atol=1e-10), (
        f"Single pixel: ndtr={val_ndtr}, dblquad={val_dblquad}"
    )


# ---- test 7: zero amplitude returns zero flux ----

def test_zero_amplitude():
    """A=0 should produce zero flux."""
    pixels = np.array([[5, 5], [6, 5], [5, 6]])
    val = flux_ndtr(pixels, 5.0, 5.0, 0.75, 0.0)
    assert val == 0.0, f"Expected 0.0, got {val}"


# ---- test 8: star far from aperture gives negligible flux ----

def test_distant_star():
    """A star far from the aperture has negligible flux in the aperture."""
    sigma = 0.75
    A = 1.0

    pixels = np.array([
        [x, y]
        for x in range(4, 7)
        for y in range(4, 7)
    ])

    # Star at (50, 50) -- very far away
    val = flux_ndtr(pixels, 50.0, 50.0, sigma, A)
    assert val < 1e-100, f"Expected negligible flux, got {val}"


# ---- test 9: source-level regression -- calc_depths uses ndtr, not dblquad ----

def test_calc_depths_uses_ndtr():
    """Verify that target.calc_depths source uses ndtr, not dblquad.

    This is a regression test: if someone reverts to the dblquad
    implementation, this test will fail. Without this, the other tests
    in this file (which exercise standalone helpers, not package code)
    would pass on both the fixed and unfixed branches.
    """
    # Import from the repo, not from an installed package
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from triceratops.triceratops import target
    source = inspect.getsource(target.calc_depths)

    # Must use ndtr
    assert "ndtr(" in source, (
        "calc_depths must use ndtr() for the analytic PSF integral"
    )
    # Must NOT call dblquad
    assert "dblquad(" not in source, (
        "calc_depths must not call dblquad() -- "
        "replaced by the analytic ndtr integral"
    )
    # Must NOT reference Gauss2D
    assert "Gauss2D" not in source, (
        "calc_depths must not reference Gauss2D -- "
        "no longer needed with the analytic integral"
    )


def test_triceratops_module_imports_ndtr_not_dblquad():
    """Verify the module-level imports use ndtr, not dblquad."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import triceratops.triceratops as tr_mod
    source = inspect.getsource(tr_mod)

    # Module should import ndtr
    assert "from scipy.special import ndtr" in source, (
        "triceratops.py must import ndtr from scipy.special"
    )
    # Module should NOT import dblquad
    assert "from scipy.integrate import dblquad" not in source, (
        "triceratops.py must not import dblquad (no longer used)"
    )
