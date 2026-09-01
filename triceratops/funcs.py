import os
import numpy as np
from pandas import read_csv
from pandas.errors import EmptyDataError
from astropy import constants
from scipy.interpolate import InterpolatedUnivariateSpline
from mechanicalsoup import StatefulBrowser
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
from time import sleep
from astropy.io import fits
import ssl

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi

Mass_nodes_Torres = np.array([
    0.26, 0.47, 0.59, 0.69, 0.87, 0.98, 1.085,
    1.4, 1.65, 2.0, 2.5, 3.0, 4.4, 15.0, 40.0
    ])
Teff_nodes_Torres = np.array([
    3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650,
    7300, 8180, 9790, 11400, 15200, 30000, 42000
    ])
Rad_nodes_Torres = np.array([
    0.28, 0.47, 0.60, 0.72, 0.9, 1.05, 1.2, 1.55,
    1.8, 2.1, 2.4, 2.6, 3.0, 6.2, 11.0
    ])
Teff_spline_Torres = InterpolatedUnivariateSpline(
    Mass_nodes_Torres, Teff_nodes_Torres
    )
Rad_spline_Torres = InterpolatedUnivariateSpline(
    Mass_nodes_Torres, Rad_nodes_Torres
    )
Mass_nodes_cdwrf = np.array([
    0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63
    ])
Teff_nodes_cdwrf = np.array([
    2800, 3000, 3200, 3400, 3600, 3800, 4000
    ])
Rad_nodes_cdwrf = np.array([
    0.12, 0.165, 0.23, 0.36, 0.48, 0.585, 0.6
    ])
Teff_spline_cdwrf = InterpolatedUnivariateSpline(
    Mass_nodes_cdwrf, Teff_nodes_cdwrf
    )
Rad_spline_cdwrf = InterpolatedUnivariateSpline(
    Mass_nodes_cdwrf, Rad_nodes_cdwrf
    )


def stellar_relations(Masses: np.array,
                      max_Radii: np.array,
                      max_Teffs: np.array):
    """
    Estimates radii and effective temperatures of stars given masses.
    Args:
        Masses (numpy array): Star masses [Solar masses].
    Returns:
        Rad (numpy array): Star radii [Solar radii].
        Teff (numpy array): Star effective temperatures [K].
    """
    Radii = np.zeros(len(Masses))
    Teffs = np.zeros(len(Masses))
    mask_hot = Masses > 0.63
    mask_cool = Masses <= 0.63

    Radii[mask_hot] = Rad_spline_Torres(Masses[mask_hot])
    Teffs[mask_hot] = Teff_spline_Torres(Masses[mask_hot])
    Radii[mask_cool] = Rad_spline_cdwrf(Masses[mask_cool])
    Teffs[mask_cool] = Teff_spline_cdwrf(Masses[mask_cool])
    # don't allow estimated radii/Teffs to be above/below max/min value
    Radii[Radii > max_Radii] = max_Radii[Radii > max_Radii]
    Teffs[Teffs > max_Teffs] = max_Teffs[Teffs > max_Teffs]
    Radii[Radii < 0.1] = 0.1
    Teffs[Teffs < 2800] = 2800
    return Radii, Teffs

Mass_nodes = np.array([
    0.1, 0.15, 0.23, 0.4, 0.58, 0.7, 0.9, 1.15, 1.45, 2.2, 2.8
    ])
flux_nodes = np.array([
    -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    ])
flux_spline = InterpolatedUnivariateSpline(
    Mass_nodes, flux_nodes
    )

Mass_nodes_J = np.array([
    0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3
    ])
flux_nodes_J = np.array([
    -5.7, -3.8, -1.6, 0, 1.2, 2.9, 3.3, 4, 6
    ])/2.5
flux_spline_J = InterpolatedUnivariateSpline(
    Mass_nodes_J, flux_nodes_J
    )

Mass_nodes_H = np.array([
    0.1, 0.23, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3
    ])
flux_nodes_H = np.array([
    -4.9, -2.8, -0.9, 0.6, 1.5, 3, 3.3, 4, 6
    ])/2.5
flux_spline_H = InterpolatedUnivariateSpline(
    Mass_nodes_H, flux_nodes_H
    )

Mass_nodes_K = np.array([
    0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3
    ])
flux_nodes_K = np.array([
    -4.7, -2.9, -1.7, -0.7, 0.6, 1.6, 3, 3.3, 4, 6
    ])/2.5
flux_spline_K = InterpolatedUnivariateSpline(
    Mass_nodes_K, flux_nodes_K
    )

def flux_relation(Masses: np.array, filt: str = "TESS"):
    """
    Estimates fluxes of stars given masses.
    Args:
        Masses (numpy array): Star masses [Solar masses].
        filt (string): Photometric filter. Options are
                       TESS, Vis, J, H, and K.
    Returns:
        fluxes (numpy array): Flux ratio between star and
                              a ~1 Solar mass star.
    """
    if (filt == "TESS") or (filt == "Vis"):
        fluxes = 10**flux_spline(Masses)
    if filt == "J":
        fluxes = 10**flux_spline_J(Masses)
    if filt == "H":
        fluxes = 10**flux_spline_H(Masses)
    if filt == "K":
        fluxes = 10**flux_spline_K(Masses)
    return fluxes


def color_Teff_relations(V, Ks):
    """
    Estimates stellar effective temperature based on photometry.
    Args:
        V (float): V magnitude of star
        Ks (float): Ks magnitude of star.
    Returns:
        Teff (float): Star effective temperature [K].
    """
    if V-Ks < 5.05:
        theta = (0.54042 + 0.23676*(V-Ks) - 0.00796*(V-Ks)**2)
        Teff = 5040/theta
    elif V-Ks > 5.05:
        theta = (
            -0.4809 + 0.8009*(V-Ks)
            - 0.1039*(V-Ks)**2 + 0.0056*(V-Ks)**3
            )
        Teff = 5040/theta + 205.26
    return Teff


# mean main-sequence (dwarf) stellar sequence of Pecaut & Mamajek (2013),
# used to estimate stellar parameters of stars that are missing them in
# the TIC. See triceratops/data/mamajek_dwarf_sequence.csv for provenance.
mamajek_sequence = read_csv(
    os.path.join(
        os.path.dirname(__file__), "data", "mamajek_dwarf_sequence.csv"
        ),
    comment="#"
    )

# extinction ratios A_X / E(B-V) for R_V = 3.1 (Cardelli et al. 1989);
# 2MASS values from Indebetouw et al. (2005), Gaia from Casagrande &
# VandenBerg (2018). Only used to deredden colors, so approximate values
# are sufficient at the low reddening typical of TESS targets.
A_EBV = {
    "V": 3.10, "G": 2.80, "BP": 3.37, "RP": 2.14,
    "J": 0.72, "H": 0.46, "K": 0.31,
    }

Mbol_sun = 4.74
Teff_sun = 5772.0


def float_or_nan(x):
    """
    Converts x to a float, returning NaN if it is not a finite number.
    Args:
        x: Value to convert.
    Returns:
        value (float): float(x), or NaN.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def mamajek_column(col: str):
    """
    Returns a column of the Mamajek dwarf sequence as a float array.
    Args:
        col (str): Column name. The synthetic color "GKs" is built from
                   the absolute magnitudes M_G and M_Ks.
    Returns:
        values (numpy array): The requested column.
    """
    if col == "GKs":
        return (mamajek_sequence["M_G"] - mamajek_sequence["M_Ks"]).values
    return mamajek_sequence[col].values.astype(float)


def monotonic_interp(x: float, xvals: np.array, yvals: np.array,
                     tol: float = 0.0):
    """
    Interpolates yvals vs xvals after sorting and dropping non-increasing
    points, so that np.interp is well defined.
    Args:
        x (float): Location to interpolate at.
        xvals, yvals (numpy arrays): Points to interpolate between.
        tol (float): Allowed distance of x beyond the tabulated range of
                     xvals before NaN is returned instead.
    Returns:
        value (float): Interpolated value, or NaN if x is out of range
                       or fewer than two usable points remain.
    """
    mask = np.isfinite(xvals) & np.isfinite(yvals)
    xvals, yvals = xvals[mask], yvals[mask]
    if len(xvals) < 2:
        return np.nan
    order = np.argsort(xvals)
    xvals, yvals = xvals[order], yvals[order]
    keep = np.concatenate(([True], np.diff(xvals) > 1e-9))
    xvals, yvals = xvals[keep], yvals[keep]
    if len(xvals) < 2 or x < xvals[0] - tol or x > xvals[-1] + tol:
        return np.nan
    return float(np.interp(x, xvals, yvals))


def color_to_teff(color_value: float, col: str):
    """
    Estimates Teff from a dereddened color using the Mamajek dwarf
    sequence. The near-infrared colors J-H and H-Ks are only used for
    cool stars, where they are single-valued.
    Args:
        color_value (float): Dereddened color.
        col (str): Mamajek sequence column for that color.
    Returns:
        Teff (float): Effective temperature [K], or NaN.
    """
    teffs = mamajek_column("Teff")
    colors = mamajek_column(col)
    if col in ("JH", "HKs"):
        cool = teffs < 5500
        teffs, colors = teffs[cool], colors[cool]
    return monotonic_interp(color_value, colors, teffs, tol=0.15)


def bc_Ks(teff: float):
    """
    Bolometric correction in the 2MASS Ks band, BC_Ks = (V-Ks) + BC_V,
    interpolated over the Mamajek dwarf sequence.
    Args:
        teff (float): Effective temperature [K].
    Returns:
        BC_Ks (float): Bolometric correction [mag], or NaN.
    """
    return monotonic_interp(
        teff, mamajek_column("Teff"),
        mamajek_column("VKs") + mamajek_column("BCv"),
        tol=200.0
        )


def estimate_stellar_parameters(Vmag: float = np.nan, Gmag: float = np.nan,
                                BPmag: float = np.nan, RPmag: float = np.nan,
                                Jmag: float = np.nan, Hmag: float = np.nan,
                                Kmag: float = np.nan, plx: float = np.nan,
                                ebv: float = 0.0, logg: float = np.nan,
                                mass: float = np.nan, rad: float = np.nan,
                                Teff: float = np.nan):
    """
    Estimates a star's mass, radius, and/or effective temperature from
    broadband photometry, assuming it lies on the main sequence.

    Teff is anchored to the best available dereddened color using the
    mean dwarf sequence of Pecaut & Mamajek (2013). The radius is then
    obtained from the parallax via the Stefan-Boltzmann law when a
    parallax and Ks magnitude are available, and from the dwarf sequence
    otherwise. The mass is read from the dwarf sequence (as a function
    of M_Ks for cool stars with a parallax, else as a function of Teff).

    Stars found to be evolved (a low surface gravity, or a radius /
    luminosity well above the main sequence at their Teff) are handled
    separately: the radius is only estimated when it can come from a
    parallax, and the mass is set to a nominal 1 M_sun, since photometry
    alone does not constrain an evolved star's mass.

    Only the parameters passed as NaN are estimated. A parameter passed
    with a finite value is returned unchanged and used as a constraint
    (e.g. a known Teff is used as the anchor instead of a color).

    Args:
        Vmag, Gmag, BPmag, RPmag, Jmag, Hmag, Kmag (float): Apparent
            magnitudes (Johnson V, Gaia G/BP/RP, 2MASS J/H/Ks). Pass NaN
            for those that are unavailable.
        plx (float): Parallax [mas].
        ebv (float): Reddening E(B-V) [mag].
        logg (float): Surface gravity [dex]. Used only to identify
            evolved stars; NaN if unknown.
        mass (float): Known mass [M_sun], or NaN to estimate.
        rad (float): Known radius [R_sun], or NaN to estimate.
        Teff (float): Known effective temperature [K], or NaN to estimate.

    Returns:
        result (dict): Keys "mass", "rad", "Teff" (estimates filled in
            where possible), "estimated" (list of the parameters that
            were newly estimated), "evolved" (bool; whether the star was
            treated as evolved), and "method" (how Teff and the radius
            were determined).
    """
    Vmag, Gmag = float_or_nan(Vmag), float_or_nan(Gmag)
    BPmag, RPmag = float_or_nan(BPmag), float_or_nan(RPmag)
    Jmag, Hmag, Kmag = float_or_nan(Jmag), float_or_nan(Hmag), float_or_nan(Kmag)
    plx, logg = float_or_nan(plx), float_or_nan(logg)
    mass, rad, Teff = float_or_nan(mass), float_or_nan(rad), float_or_nan(Teff)
    ebv = float_or_nan(ebv)
    if not np.isfinite(ebv) or ebv < 0:
        ebv = 0.0

    estimated = []
    method = {"Teff": "input", "rad": "input", "mass": "input"}

    # --- effective temperature, from the best available color ---
    if not np.isfinite(Teff):
        candidates = []
        if np.isfinite(BPmag) and np.isfinite(RPmag):
            candidates.append((
                (BPmag - RPmag) - (A_EBV["BP"] - A_EBV["RP"])*ebv,
                "BpRp", "BP-RP color"
                ))
        if np.isfinite(Vmag) and np.isfinite(Kmag):
            candidates.append((
                (Vmag - Kmag) - (A_EBV["V"] - A_EBV["K"])*ebv,
                "VKs", "V-Ks color"
                ))
        if np.isfinite(Gmag) and np.isfinite(Kmag):
            candidates.append((
                (Gmag - Kmag) - (A_EBV["G"] - A_EBV["K"])*ebv,
                "GKs", "G-Ks color"
                ))
        if np.isfinite(Gmag) and np.isfinite(RPmag):
            candidates.append((
                (Gmag - RPmag) - (A_EBV["G"] - A_EBV["RP"])*ebv,
                "GRp", "G-RP color"
                ))
        if np.isfinite(Jmag) and np.isfinite(Hmag):
            candidates.append((
                (Jmag - Hmag) - (A_EBV["J"] - A_EBV["H"])*ebv,
                "JH", "J-H color"
                ))
        if np.isfinite(Hmag) and np.isfinite(Kmag):
            candidates.append((
                (Hmag - Kmag) - (A_EBV["H"] - A_EBV["K"])*ebv,
                "HKs", "H-Ks color"
                ))
        for color_value, col, label in candidates:
            teff_try = color_to_teff(color_value, col)
            if np.isfinite(teff_try):
                Teff = float(np.clip(teff_try, 2300.0, 50000.0))
                method["Teff"] = label
                estimated.append("Teff")
                break

    # M_Ks (absolute Ks magnitude), used for the evolved-star test, the
    # Stefan-Boltzmann radius, and the cool-star mass relation
    if np.isfinite(plx) and plx > 0 and np.isfinite(Kmag):
        M_Ks = Kmag - A_EBV["K"]*ebv - 10.0 + 5.0*np.log10(plx)
    else:
        M_Ks = np.nan

    # --- is the star evolved (off the main sequence)? ---
    # flagged by a low surface gravity, a radius well above the
    # dwarf-sequence value at its Teff, or an over-luminous M_Ks
    evolved = False
    if np.isfinite(logg) and logg < 3.5:
        evolved = True
    elif np.isfinite(rad) and np.isfinite(Teff):
        rad_ms = monotonic_interp(
            Teff, mamajek_column("Teff"),
            mamajek_column("R_Rsun"), tol=200.0
            )
        if np.isfinite(rad_ms) and rad > 1.7*rad_ms:
            evolved = True
    elif np.isfinite(M_Ks) and np.isfinite(Teff):
        mks_ms = monotonic_interp(
            Teff, mamajek_column("Teff"),
            mamajek_column("M_Ks"), tol=200.0
            )
        if np.isfinite(mks_ms) and M_Ks < mks_ms - 1.0:
            evolved = True

    # --- radius ---
    if not np.isfinite(rad) and np.isfinite(Teff):
        if np.isfinite(M_Ks):
            bc = bc_Ks(Teff)
            if np.isfinite(bc):
                M_bol = M_Ks + bc
                lum = 10.0**(-0.4*(M_bol - Mbol_sun))
                rad = np.sqrt(lum) / (Teff/Teff_sun)**2
                method["rad"] = "Stefan-Boltzmann (parallax + Ks)"
        if not np.isfinite(rad) and not evolved:
            # the dwarf-sequence radius is only valid on the main sequence
            rad = monotonic_interp(
                Teff, mamajek_column("Teff"),
                mamajek_column("R_Rsun"), tol=200.0
                )
            method["rad"] = "dwarf sequence (Teff)"
        if np.isfinite(rad):
            rad = float(np.clip(rad, 0.05, 200.0))
            estimated.append("rad")

    # --- mass ---
    if not np.isfinite(mass):
        if evolved:
            # photometry does not constrain an evolved star's mass;
            # adopt 1 M_sun (near the peak of the field red-giant mass
            # distribution)
            mass = 1.0
            method["mass"] = "assumed (evolved star)"
            estimated.append("mass")
        elif np.isfinite(Teff):
            if (np.isfinite(M_Ks) and Teff < 4200.0):
                mass = monotonic_interp(
                    M_Ks, mamajek_column("M_Ks"),
                    mamajek_column("Msun"), tol=0.3
                    )
            if not np.isfinite(mass):
                mass = monotonic_interp(
                    Teff, mamajek_column("Teff"),
                    mamajek_column("Msun"), tol=200.0
                    )
            if np.isfinite(mass):
                mass = float(np.clip(mass, 0.07, 100.0))
                method["mass"] = "dwarf sequence"
                estimated.append("mass")

    return {
        "mass": mass, "rad": rad, "Teff": Teff,
        "estimated": estimated, "evolved": evolved, "method": method,
        }


def renorm_flux(flux, flux_err, star_fluxratio: float):
    """
    Renormalizes light curve flux to account for flux contribution
    due to nearby stars.
    Args:
        flux (numpy array): Normalized flux of each data point.
        star_fluxratio (float): Proportion of flux that comes
                                from the star.
    Returns:
        renormed_flux (nump array): Remormalized flux of each point.
    """
    renormed_flux = (flux - (1 - star_fluxratio)) / star_fluxratio
    renormed_flux_err = flux_err / star_fluxratio
    return renormed_flux, renormed_flux_err


def Gauss2D(x, y, mu_x, mu_y, sigma, A):
    """
    Calculates a circular Gaussian at specified grid points.
    Args:
        x, y (1D numpy arrays): Grid that you would like to calculate
                                Gaussian over.
        mu_x, mu_y (floats): Locations of star / Gaussian peak.
        sigma (float): Standard deviation of Gaussian
        A (float): Area under Gaussian.
    Returns:
    """
    # dblquad passes scalar x/y; np.meshgrid on scalars returns 0-d arrays
    # on NumPy 2.x, which scipy cannot convert internally. Fast-path for
    # the scalar case (the only case dblquad uses).
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        x0, y0 = float(x), float(y)
        exponent = ((x0 - mu_x)**2 + (y0 - mu_y)**2) / (2*sigma**2)
        return float(A / (2*np.pi*sigma**2) * np.exp(-exponent))
    xgrid, ygrid = np.meshgrid(x, y)
    exponent = ((xgrid-mu_x)**2 + (ygrid-mu_y)**2)/(2*sigma**2)
    return A/(2*np.pi*sigma**2)*np.exp(-exponent)


def file_to_contrast_curve(contrast_curve_file: str):
    """
    Obtains arrays of contrast and separation from a
    contrast curve file.
    Args:
        contrast_curve_file (str): Path to contrast curve text file.
                                   File should contain column with
                                   separations (in arcsec)
                                   followed by column with Delta_mags.
    Returns:
        separations (numpy array): Separation at contrast (arcsec).
        contrasts (numpy array): Contrast at separation (delta_mag).
    """
    data = np.loadtxt(contrast_curve_file, delimiter=',')
    separations = data.T[0]
    contrasts = np.abs(data.T[1])
    return separations, contrasts


def separation_at_contrast(delta_mags: np.array,
                           separations: np.array,
                           contrasts: np.array):
    """
    Calculates the limiting separation (in arcsecs)
    at a given delta_mag.
    Args:
        delta_mag (numpy array): Contrasts of simulated
                                 companions (delta_mag).
        separations (numpy array): Separation at contrast (arcsec).
        contrasts (numpy array): Contrast at separation (delta_mag).
    Returns:
        sep (numpy array): Separation beyond which we can rule out
                           the simulated companion (arcsec).
    """
    sep = np.interp(delta_mags, contrasts, separations)
    return sep


def parse_contrast_curves(contrast_curve_file, filt="TESS"):
    """
    Normalizes contrast curve inputs so that one or more contrast curves
    can be supplied to the analysis.
    Args:
        contrast_curve_file (str or list of str): Path(s) to contrast
            curve text file(s), or None.
        filt (str or list of str): Photometric filter(s) of the contrast
            curve(s). A single filter is broadcast to all files. Options
            are TESS, Vis, J, H, and K.
    Returns:
        files (list of str or None): List of contrast curve paths, or
            None if no contrast curve was provided.
        filts (list of str or None): Matching list of filters, or None.
    """
    if contrast_curve_file is None:
        return None, None
    if (isinstance(contrast_curve_file, (str, bytes))
            or not hasattr(contrast_curve_file, "__iter__")):
        files = [contrast_curve_file]
    else:
        files = list(contrast_curve_file)
    if len(files) == 0:
        return None, None
    if isinstance(filt, (str, bytes)) or not hasattr(filt, "__iter__"):
        filts = [filt]
    else:
        filts = list(filt)
    if len(filts) == 1:
        filts = filts*len(files)
    if len(filts) != len(files):
        raise ValueError(
            "Number of contrast curve filters ({0}) does not match the "
            "number of contrast curve files ({1}).".format(
                len(filts), len(files)
                )
            )
    return files, filts


def limiting_separation(delta_mags_list: list, contrast_curve_files: list):
    """
    Determines the limiting angular separation (in arcsec) beyond which
    each simulated companion can be ruled out, combining the constraints
    from one or more contrast curves. A companion is considered ruled out
    if any single contrast curve rules it out, so the tightest (smallest)
    limiting separation across all curves is adopted.
    Args:
        delta_mags_list (list of numpy arrays): Contrasts of the simulated
            companions (delta_mag), one array per contrast curve, each
            evaluated in that curve's photometric filter.
        contrast_curve_files (list of str): Paths to the contrast curve
            text files, in the same order as delta_mags_list.
    Returns:
        seps (numpy array): Separation beyond which each simulated
            companion can be ruled out (arcsec).
    """
    seps = None
    for delta_mags, cc_file in zip(delta_mags_list, contrast_curve_files):
        separations, contrasts = file_to_contrast_curve(cc_file)
        this_seps = separation_at_contrast(
            delta_mags, separations, contrasts
            )
        if seps is None:
            seps = this_seps
        else:
            seps = np.minimum(seps, this_seps)
    return seps


def query_TRILEGAL(RA: float, Dec: float, verbose: int = 1, verify_ssl: bool = True):
    """
    Begins TRILEGAL query.
    Args:
        RA, Dec: Coordinates of the target.
        verbose: 1 to print progress, 0 to print nothing.
        verify_ssl: True to verify SSL certificates, False to ignore.
                    ONLY SET TO FALSE IF ABSOLUTELY NECESSARY.
    Returns:
        output_url (str): URL of page with query results.
    """
    for version in ("1.6", "1.5"):
        url = f"https://stev.oapd.inaf.it/cgi-bin/trilegal_{version}"
        try:
            # TRILEGAL's legacy HTML is malformed enough that MechanicalSoup's
            # default lxml parser can fail to see its form at all.  Use the
            # stdlib parser, which is more tolerant of that response.
            browser = StatefulBrowser(soup_config={"features": "html.parser"})
            browser.session.verify = verify_ssl
            browser.set_user_agent(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            )
            browser.open(url, timeout=60)
            browser.select_form(nr=0)
            browser["gal_coord"] = "2"
            browser["eq_alpha"] = str(RA)
            browser["eq_delta"] = str(Dec)
            browser["field"] = "0.1"
            if version == "1.6":
                browser["photsys_file"] = "tab_mag_odfnew/tab_mag_TESS_2mass.dat"
            else:
                browser["photsys_file"] = "tab_mag_odfnew/tab_mag_2mass.dat"
            browser["icm_lim"] = "1"
            browser["mag_lim"] = "21"
            browser["binary_kind"] = "0"
            browser.submit_selected()
            if verbose:
                print(f"TRILEGAL v{version} form submitted.")
            sleep(5)
            links = browser.get_current_page().select("a")
            if len(links) > 0:
                data_link = links[0].get("href")
                return f"https://stev.oapd.inaf.it/{data_link[3:]}"
            if verbose:
                print(f"TRILEGAL v{version}: no result links, trying next...")
        except Exception as exc:
            if verbose:
                print(f"TRILEGAL v{version} failed: {exc}")
            continue
    print(
        "TRILEGAL unavailable after trying versions 1.6 and 1.5. "
        "The scenarios that need a simulated background population "
        "(DTP, DEB, DEBx2P, BTP, BEB, BEBx2P) will be skipped."
    )
    return None


def save_trilegal(output_url, ID: int):
    """
    Saves results of trilegal query to a csv.
    Args:
        output_url (str): URL of page with query results.
        ID (int): ID of the target.
    Returns:
        fname (str): File name of csv containing trilegal results. 
    """
    if output_url is None:
        print(
            "Could not access TRILEGAL. "
            + "Ignoring BTP, BEB, BEBx2P, DTP, DEB, and DEBx2P scenarios."
            )
        return 0.0
    else:
        for i in range(5):
            try:
                last = read_csv(output_url, header=None)[-1:]
            except EmptyDataError:
                if i < 4:
                    print("...")
                    sleep(10)
                    continue
                raise
            if last.values[0, 0] != "#TRILEGAL normally terminated":
                print("...")
                sleep(10)
            elif last.values[0, 0] == "#TRILEGAL normally terminated":
                break
        df = read_csv(output_url, delim_whitespace=True)
        fname = str(ID) + "_TRILEGAL.csv"
        df.to_csv(fname)
        return fname

def trilegal_results(trilegal_fname: str, Tmag: float):
    """
    Retrieves arrays of stars from trilegal query.
    Args:
        trilegal_fname (str): File containing query results.
        Tmag (float): TESS magnitude of the star.
    Returns:
        Tmags (numpy array): TESS magnitude of all stars
                             fainter than the target.
        Masses (numpy array): Masses of all stars fainter than the
                              target [Solar masses].
        loggs (numpy array): loggs of all stars fainter than the
                             target [log10(cm/s^2)].
        Teffs (numpy array): Teffs of all stars fainter than the
                             target [K].
        Zs (numpy array): Metallicities of all stars fainter than the
                          target [dex].
    """
    df = read_csv(trilegal_fname)[:-2]
    Masses = df["Mact"].values
    loggs = df["logg"].values
    Teffs = 10**df["logTe"].values
    Zs = np.array(df["[M/H]"], dtype=float)
    Tmags = df["TESS"].values
    Jmags = df["J"].values
    Hmags = df["H"].values
    Kmags = df["Ks"].values
    headers = np.array(list(df))
    # if able to use TRILEGAL v1.6 and get TESS mags, use them
    if "TESS" in headers:
        mask = (Tmags >= Tmag)
        Masses = Masses[mask]
        loggs = loggs[mask]
        Teffs = Teffs[mask]
        Zs = Zs[mask]
        Tmags = Tmags[mask]
        Jmags = Jmags[mask]
        Hmags = Hmags[mask]
        Kmags = Kmags[mask]
    # otherwise, use 2mass mags from TRILEGAL v1.5 and convert
    # to T mags using the relations from section 2.2.1.1 of
    # Stassun et al. 2018
    else:
        Tmags = np.zeros(df.shape[0])
        for i, (J, Ks) in enumerate(zip(Jmags, Kmags)):
            if (-0.1 <= J-Ks <= 0.70):
                Tmags[i] = (
                    J + 1.22163*(J-Ks)**3
                    - 1.74299*(J-Ks)**2 + 1.89115*(J-Ks) + 0.0563
                    )
            elif (0.7 < J-Ks <= 1.0):
                Tmags[i] = (
                    J - 269.372*(J-Ks)**3
                    + 668.453*(J-Ks)**2 - 545.64*(J-Ks) + 147.811
                    )
            elif (J-Ks < -0.1):
                Tmags[i] = J + 0.5
            elif (J-Ks > 1.0):
                Tmags[i] = J + 1.75
        mask = (Tmags >= Tmag)
        Masses = Masses[mask]
        loggs = loggs[mask]
        Teffs = Teffs[mask]
        Zs = Zs[mask]
        Tmags = Tmags[mask]
        Jmags = Jmags[mask]
        Hmags = Hmags[mask]
        Kmags = Kmags[mask]
    return Tmags, Masses, loggs, Teffs, Zs, Jmags, Hmags, Kmags
def segment_ID(str_segment):
    """
    Returns TIC ID with appropriate number of leading zeros
    for MAST querying.
    """
    if len(str_segment) == 0:
        return "0000"
    elif len(str_segment) == 1:
        return "000"+str_segment
    elif len(str_segment) == 2:
        return "00"+str_segment
    elif len(str_segment) == 3:
        return "0"+str_segment
    elif len(str_segment) == 4:
        return str_segment

        
def find_url(ID: str, sector: int):
    """
    Returns url of FITS file for TESS SPOC lc.
    Within this file is the aperture used in the sector.
    Args:
        ID (str): TIC ID of star.
        sector (int): TESS sector.
    Returns:
        url (str): FITS file url.
    """
    url = "https://archive.stsci.edu/missions/tess/tid/"
    
    if len(str(sector)) == 1:
        str1 = "s000"+str(sector)
    elif len(str(sector)) == 2:
        str1 = "s00"+str(sector)
    elif len(str(sector)) == 3:
        str1 = "s0"+str(sector)
    else:
        raise ValueError("TESS sector must be a positive integer with at most three digits")
           
    str2 = segment_ID(str(ID)[-16:-12])
    str3 = segment_ID(str(ID)[-12:-8])
    str4 = segment_ID(str(ID)[-8:-4])
    str5 = segment_ID(str(ID)[-4:])
        
    url += str1+"/"+str2+"/"+str3+"/"+str4+"/"+str5+"/"

    no_data_msg = (
        "No SPOC light curve found for TIC " + str(ID) + " in sector "
        + str(sector) + ", so no aperture is available. This target may "
        + "not have 2-minute cadence data in that sector."
        )
    try:
        urlpath = urlopen(url)
        string = urlpath.read().decode('utf-8')
    except HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(no_data_msg) from e
        raise RuntimeError(
            "MAST archive returned an error for TIC " + str(ID)
            + " sector " + str(sector) + " (" + url + ")."
            ) from e
    except URLError as e:
        raise RuntimeError(
            "Could not reach the MAST archive for TIC " + str(ID)
            + " sector " + str(sector) + " (" + url + ")."
            ) from e
    soup = BeautifulSoup(string, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href') or ""
        if href[-9:] == "s_lc.fits":
            return url + href

    raise FileNotFoundError(no_data_msg)


def get_aperture(ID, sector):
    """
    Returns the SPOC optimal photometric aperture for a given sector, in
    the [[col, row], ...] format expected by target.calc_depths().
    Args:
        ID (str): TIC ID of star.
        sector (int): TESS sector.
    Returns:
        ap_pixels (numpy array): Aperture pixels, one [col, row] per row.
    """
    fits_file = find_url(ID, sector)

    with fits.open(fits_file, mode="readonly") as hdulist:
        aperture = hdulist["APERTURE"].data.astype(int)
        col_ref = hdulist["APERTURE"].header["CRVAL1P"]
        row_ref = hdulist["APERTURE"].header["CRVAL2P"]

    # bit 2 of the SPOC aperture bitmask flags pixels used in the
    # optimal photometric aperture
    rows, cols = np.nonzero(np.bitwise_and(aperture, 2))
    if len(rows) == 0:
        raise ValueError(
            "The SPOC aperture mask for TIC " + str(ID) + " sector "
            + str(sector) + " contains no optimal-aperture pixels."
            )
    ap_pixels = np.column_stack([cols + col_ref, rows + row_ref])

    return ap_pixels
