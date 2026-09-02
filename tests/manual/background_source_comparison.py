"""Compare the Gaia and TRILEGAL background-star populations, and their
effect on the FPP, for a few targets.

For each target it:
  - builds the Gaia and TRILEGAL field-star populations and prints the
    star count, magnitude and mass distributions, and the resulting
    background-star prior;
  - optionally (``--fpp``) runs calc_probs() both ways and prints FPP,
    NFPP and the blended-star (D/B) scenario probabilities.

This hits the Gaia archive and the TRILEGAL web form (and, with
``--fpp``, MAST/TESSCut), so it is not part of the pytest suite.

    python tests/manual/background_source_comparison.py
    python tests/manual/background_source_comparison.py --fpp
"""
import argparse
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_LC = os.path.join(_REPO_ROOT, "examples", "TOI465_01_lightcurve.csv")

# (name, TIC ID, sectors, RA, Dec, P_orb, tdepth) -- a spread in galactic
# latitude; WASP-156 is the only one with a bundled light curve, so it is
# the only one used for the --fpp comparison
TARGETS = [
    ("TOI 465 / WASP-156", 270380593, [4], 32.781765, 2.418021,
     3.836169, 0.005),
    ("mid-latitude field", None, None, 120.0, 20.0, None, None),
    ("near-plane field", None, None, 279.0, -4.0, None, None),
]


def population_summary(name, ra, dec):
    from triceratops.funcs import (query_gaia_background, query_TRILEGAL,
                                   save_trilegal, trilegal_results)
    from triceratops.priors import lnprior_background

    b = SkyCoord(ra, dec, unit="deg").galactic.b.deg
    seps = np.array([0.5, 1.0, 2.0])  # arcsec
    print("\n=== {0}  (b = {1:+.1f} deg) ===".format(name, b))

    def describe(label, fname):
        if not isinstance(fname, str):
            print("  {0:9s}: unavailable".format(label))
            return None
        T, M, lg, Te, Z, J, H, K = trilegal_results(fname, 0.0)
        lp = lnprior_background(len(T), seps)
        print("  {0:9s}: N={1:6d}  T(med/p95)={2:.1f}/{3:.1f}  "
              "mass(med)={4:.2f}  logg(med)={5:.2f}  "
              "lnprior_bg={6}".format(
                  label, len(T), np.median(T), np.percentile(T, 95),
                  np.median(M), np.median(lg), np.round(lp, 1).tolist()))
        return len(T)

    n_gaia = describe("gaia", query_gaia_background(ra, dec, 0, verbose=0))
    try:
        n_tri = describe(
            "trilegal", save_trilegal(query_TRILEGAL(ra, dec, verbose=0), 0))
    except Exception as exc:
        print("  trilegal: error ({0})".format(exc))
        n_tri = None
    if n_gaia and n_tri:
        r = n_gaia/n_tri
        print("  N_gaia / N_trilegal = {0:.2f}  "
              "(background-star prior shifts by {1:+.2f} nat)".format(
                  r, np.log(r)))


def fpp_comparison():
    import triceratops.triceratops as tr

    name, ID, sectors, ra, dec, P_orb, tdepth = TARGETS[0]
    time, flux, flux_err = np.loadtxt(_LC, delimiter=",", unpack=True)
    ferr = float(np.mean(flux_err))
    print("\n=== FPP comparison: {0} ===".format(name))
    for source in ("gaia", "trilegal"):
        try:
            t = tr.target(ID=ID, sectors=np.array(sectors),
                          background_population_source=source)
            t.calc_depths(tdepth=tdepth,
                          all_ap_pixels=t.get_spoc_apertures())
            t.calc_probs(time=time, flux_0=flux, flux_err_0=ferr,
                         P_orb=P_orb, N=20000, parallel=False, verbose=0)
            db = t.probs[t.probs["scenario"].isin(
                ["DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P"])]
            print("  {0:9s}: FPP={1:.4f}  NFPP={2:.4f}  "
                  "P(D/B scenarios)={3:.4f}".format(
                      source, t.FPP, t.NFPP, db["prob"].sum()))
        except Exception as exc:
            print("  {0:9s}: error ({1})".format(source, exc))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fpp", action="store_true",
                   help="also run the full calc_probs FPP comparison")
    args = p.parse_args()
    for name, ID, sectors, ra, dec, P_orb, tdepth in TARGETS:
        population_summary(name, ra, dec)
    if args.fpp:
        fpp_comparison()
