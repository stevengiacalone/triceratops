import numpy as np
from pandas import read_csv
from astropy import constants
from scipy.interpolate import InterpolatedUnivariateSpline
from mechanicalsoup import StatefulBrowser
from time import sleep

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

def flux_relation(Masses: np.array, band: str = "TESS"):
    """
    Estimates fluxes of stars given masses.
    Args:
        Masses (numpy array): Star masses [Solar masses].
        band (string): Photometric band. Options are
                       TESS, Vis, J, H, and K.
    Returns:
        fluxes (numpy array): Flux ratio between star and
                              a ~1 Solar mass star.
    """
    if (band == "TESS") or (band == "Vis"):
        fluxes = 10**flux_spline(Masses)
    if band == "J":
        fluxes = 10**flux_spline_J(Masses)
    if band == "H":
        fluxes = 10**flux_spline_H(Masses)
    if band == "K":
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
    xgrid, ygrid = np.meshgrid(x, y)
    exponent = ((xgrid-mu_x)**2 + (ygrid-mu_y)**2)/(2*sigma**2)
    GaussExp = np.exp(-exponent)
    return A/(2*np.pi*sigma**2)*GaussExp


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


def query_TRILEGAL(RA: float, Dec: float, verbose: int = 1):
    """
    Begins TRILEGAL query.
    Args:
        RA, Dec: Coordinates of the target.
        verbose: 1 to print progress, 0 to print nothing.
    Returns:
        output_url (str): URL of page with query results.
    """
    # fill out and submit online TRILEGAL form
    browser = StatefulBrowser()
    browser.open("http://stev.oapd.inaf.it/cgi-bin/trilegal_1.6")
    browser.select_form(nr=0)
    browser["gal_coord"] = "2"
    browser["eq_alpha"] = str(RA)
    browser["eq_delta"] = str(Dec)
    browser["field"] = "0.1"
    browser["photsys_file"] = "tab_mag_odfnew/tab_mag_TESS_2mass.dat"
    browser["icm_lim"] = "1"
    browser["mag_lim"] = "21"
    browser["binary_kind"] = "0"
    browser.submit_selected()
    if verbose == 1:
        print("TRILEGAL form submitted.")
    sleep(5)
    if len(browser.get_current_page().select("a")) == 0:
        browser = StatefulBrowser()
        browser.open("http://stev.oapd.inaf.it/cgi-bin/trilegal_1.5")
        browser.select_form(nr=0)
        browser["gal_coord"] = "2"
        browser["eq_alpha"] = str(RA)
        browser["eq_delta"] = str(Dec)
        browser["field"] = "0.1"
        browser["photsys_file"] = "tab_mag_odfnew/tab_mag_2mass.dat"
        browser["icm_lim"] = "1"
        browser["mag_lim"] = "21"
        browser["binary_kind"] = "0"
        browser.submit_selected()
        # print("TRILEGAL form submitted.")
        sleep(5)
        if len(browser.get_current_page().select("a")) == 0:
            print(
                "TRILEGAL too busy, \
                using saved stellar populations instead."
                )
            return None
        else:
            this_page = browser.get_current_page()
            data_link = this_page.select("a")[0].get("href")
            output_url = "http://stev.oapd.inaf.it/"+data_link[3:]
            return output_url
    else:
        this_page = browser.get_current_page()
        data_link = this_page.select("a")[0].get("href")
        output_url = "http://stev.oapd.inaf.it/"+data_link[3:]
        return output_url


def trilegal_results(output_url, Tmag: float):
    """
    Retrieves arrays of stars from trilegal query.
    Args:
        output_url (str): URL of page with query results.
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
    if output_url is None:
        print(
            "Could not access TRILEGAL. "
            + "Ignoring BTP, BEB, BEBx2P, DTP, DEB, and DEBx2P scenarios."
            )
        return 0.0
    else:
        for i in range(1000):
            last = read_csv(output_url, header=None)[-1:]
            if last.values[0, 0] != "#TRILEGAL normally terminated":
                print("...")
                sleep(10)
            elif last.values[0, 0] == "#TRILEGAL normally terminated":
                break
        # find number of stars fainter than target
        df = read_csv(output_url, delim_whitespace=True)[:-2]
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