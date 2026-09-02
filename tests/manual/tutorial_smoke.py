"""End-to-end smoke test of the first tutorial example.

Runs the TESS example from docs/tutorials/example.ipynb (WASP-156b /
TIC 270380593, sector 4), with the default Gaia DR3 background-star
population, a few times, and checks that calc_probs() returns finite
FPP / NFPP in [0, 1]. Prints the mean and standard deviation.

This hits MAST (TIC, TESSCut) and the Gaia archive, so it is not part
of the pytest suite; it is run as a separate, non-blocking CI job and
can be run by hand:

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
N_DRAWS = 20000
N_REPEATS = 6


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

    print("background population source: %s"
          % getattr(target, "background_population_source", "?"))
    print("calc_probs (N=%d, x%d) ..." % (N_DRAWS, N_REPEATS))
    ferr = float(np.mean(flux_err))
    fpps, nfpps = [], []
    for k in range(N_REPEATS):
        target.calc_probs(
            time=time, flux_0=flux, flux_err_0=ferr,
            P_orb=P_ORB, N=N_DRAWS, parallel=False, verbose=0,
        )
        fpps.append(target.FPP)
        nfpps.append(target.NFPP)
        print("  run %d: FPP = %.4f  NFPP = %.4f"
              % (k + 1, target.FPP, target.NFPP))
    fpps, nfpps = np.array(fpps), np.array(nfpps)

    ok = np.all(np.isfinite(fpps)) and np.all(np.isfinite(nfpps)) \
        and np.all((fpps >= 0) & (fpps <= 1)) \
        and np.all((nfpps >= 0) & (nfpps <= 1))

    line = ("WASP-156b (Gaia population): "
            "FPP = %.4f +/- %.4f, NFPP = %.4f +/- %.4f"
            % (fpps.mean(), fpps.std(), nfpps.mean(), nfpps.std()))
    print(line)
    # surface the numbers as a GitHub Actions notice annotation
    if os.environ.get("GITHUB_ACTIONS"):
        print("::notice title=Tutorial FPP/NFPP::" + line)

    if not ok:
        raise RuntimeError("FPP/NFPP not all finite values in [0, 1]")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
