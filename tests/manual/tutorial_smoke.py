"""End-to-end smoke test of the first tutorial example.

Runs the TESS example from docs/tutorials/example.ipynb (WASP-156b /
TIC 270380593, sector 4) with a small number of Monte Carlo draws and
checks that calc_probs() returns finite FPP / NFPP in [0, 1].

This hits MAST (TIC, TESSCut) and TRILEGAL, so it is not part of the
pytest suite; it is run as a separate, non-blocking CI job and can be
run by hand:

    python tests/manual/tutorial_smoke.py
"""
import os
import sys
import traceback

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_LC = os.path.join(_REPO_ROOT, "examples", "TOI465_01_lightcurve.csv")

ID = 270380593
SECTORS = np.array([4])
P_ORB = 3.836169
TDEPTH = 0.005
N_DRAWS = 5000


def bin_light_curve(time, flux, flux_err, n_bins=200):
    edges = np.linspace(time.min(), time.max(), n_bins + 1)
    idx = np.clip(np.digitize(time, edges) - 1, 0, n_bins - 1)
    t, f, e = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        t.append(np.mean(time[m]))
        f.append(np.mean(flux[m]))
        e.append(np.mean(flux_err[m]) / np.sqrt(np.sum(m)))
    return np.array(t), np.array(f), np.array(e)


def main():
    import triceratops.triceratops as tr

    print("triceratops", getattr(tr, "__version__", "?"),
          "| python", sys.version.split()[0], "| numpy", np.__version__)

    print("building target ...")
    target = tr.target(ID=ID, sectors=SECTORS)
    print("  .stars: %d rows" % len(target.stars))

    print("fetching SPOC aperture ...")
    apertures = target.get_spoc_apertures()
    if apertures[0] is None:
        raise RuntimeError("no SPOC aperture returned for sector 4")

    print("calc_depths ...")
    target.calc_depths(tdepth=TDEPTH, all_ap_pixels=apertures)

    time, flux, flux_err = np.loadtxt(_LC, delimiter=",", unpack=True)
    time, flux, flux_err = bin_light_curve(time, flux, flux_err)
    print("  binned light curve: %d points" % time.size)

    print("calc_probs (N=%d) ..." % N_DRAWS)
    target.calc_probs(
        time=time, flux_0=flux, flux_err_0=float(np.mean(flux_err)),
        P_orb=P_ORB, N=N_DRAWS, parallel=False, verbose=0,
    )
    fpp, nfpp = target.FPP, target.NFPP
    print("  FPP  = %.4f" % fpp)
    print("  NFPP = %.4f" % nfpp)

    ok = True
    for name, val in (("FPP", fpp), ("NFPP", nfpp)):
        if not np.isfinite(val) or not (0.0 <= val <= 1.0):
            print("  FAIL: %s = %r is not a finite value in [0, 1]"
                  % (name, val))
            ok = False
    if not ok:
        raise RuntimeError("FPP/NFPP out of range")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
