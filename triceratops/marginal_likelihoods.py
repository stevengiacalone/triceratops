import numpy as np
from pandas import read_csv
from astropy import constants
from pkg_resources import resource_filename

from .likelihoods import *
from .priors import *
from .funcs import stellar_relations, flux_relation

np.seterr(divide='ignore')

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi
ln2pi = np.log(2*pi)

# load TESS limb darkening coefficients
LDC_FILE = resource_filename('triceratops', 'data/ldc_tess.csv')
ldc_T = read_csv(LDC_FILE)
ldc_T_Zs = np.array(ldc_T.Z, dtype=float)
ldc_T_Teffs = np.array(ldc_T.Teff, dtype=int)
ldc_T_loggs = np.array(ldc_T.logg, dtype=float)
ldc_T_u1s = np.array(ldc_T.aLSM, dtype=float)
ldc_T_u2s = np.array(ldc_T.bLSM, dtype=float)

# load Kepler limb darkening coefficients
LDC_FILE = resource_filename('triceratops', 'data/ldc_kepler.csv')
ldc_K = read_csv(LDC_FILE)
ldc_K_Zs = np.array(ldc_K.Z, dtype=float)
ldc_K_Teffs = np.array(ldc_K.Teff, dtype=int)
ldc_K_loggs = np.array(ldc_K.logg, dtype=float)
ldc_K_u1s = np.array(ldc_K.a, dtype=float)
ldc_K_u2s = np.array(ldc_K.b, dtype=float)

def lnZ_TTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the TTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    a = ((G*M_s*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # calculate short-period planet prior for star of mass M_s
    lnprior_Mstar = lnprior_Mstar_planet(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from prior distributions
    rps = sample_rp(np.random.rand(N), np.full_like(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    Ptra = (rps*Rearth + R_s*Rsun)/a * e_corr

    # find instances with collisions
    coll = ((rps*Rearth + R_s*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False)
        # calculate lnL for transiting systems
        a_arr = np.full(N, a)
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a_arr[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1.:
                inc_min = np.arccos(Ptra[i]) * 180./pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a, R_s, u1, u2,
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL + lnprior_Mstar + lnprior_Porb + 600)
        ))
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': rps[idx],
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': np.zeros(N_samples),
        'R_EB': np.zeros(N_samples),
        'fluxratio_EB': np.zeros(N_samples),
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    return res


def lnZ_TEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the TEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # sample from prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate properties of the drawn EBs
    masses = qs*M_s
    radii, Teffs = stellar_relations(
        masses, np.full(N, R_s), np.full(N, Teff)
        )
    # calculate flux ratios in the TESS band
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(np.array([M_s])))
        )

    # calculate short-period binary prior for star of mass M_s
    lnprior_Mstar = lnprior_Mstar_binary(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
    a_twin = ((G*(M_s+masses)*Msun)/(4*pi**2)*(2*P_orb*86400)**2)**(1/3)
    Ptra_twin = (radii*Rsun + R_s*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(R_s*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + R_s*Rsun) > a*(1-eccs))
    coll_twin = ((2*R_s*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95)
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95)
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        companion_fluxratio = np.zeros(N)
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] < 0.95) & (coll[i] == False):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95)
                & (coll_twin[i] == False)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, radii[i], fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL + lnprior_Mstar + lnprior_Porb + 600)
        ))
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL_twin + lnprior_Mstar + lnprior_Porb + 600)
        ))
    lnZ = np.log(Z)
    res_twin = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, 2*P_orb),
        'inc': incs[idx],
        'b': b_twin[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    return res, res_twin


def lnZ_PTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
    """
    Calculates the marginal likelihood of the PTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    a = ((G*M_s*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # sample from q prior distributions
    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1/M_s] = 0.1/M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    # calculate properties of the drawn companions
    masses_comp = qs_comp*M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
        )
    # calculate flux ratios in the TESS band
    fluxratios_comp = (
        flux_relation(masses_comp)
        / (flux_relation(masses_comp) + flux_relation(np.array([M_s])))
        )

    # calculate priors for companions
    if molusc_file is None:
        if contrast_curve_file is None:
            # use TESS/Vis band flux ratios
            delta_mags = 2.5*np.log10(
                fluxratios_comp/(1-fluxratios_comp)
                )
            lnprior_companion = lnprior_bound_TP(
                M_s, plx, np.abs(delta_mags),
                np.array([2.2]), np.array([1.0])
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # use flux ratio of contrast curve filter
            fluxratios_comp_cc = (
                flux_relation(masses_comp, filt)
                / (flux_relation(masses_comp, filt)
                    + flux_relation(np.array([M_s]), filt))
                )
            delta_mags = 2.5*np.log10(
                fluxratios_comp_cc/(1-fluxratios_comp_cc)
                )
            separations, contrasts = file_to_contrast_curve(
                contrast_curve_file
                )
            lnprior_companion = lnprior_bound_TP(
                M_s, plx, np.abs(delta_mags), separations, contrasts
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        lnprior_companion = np.zeros(N)


    # calculate short-period planet prior for star of mass M_s
    lnprior_Mstar = lnprior_Mstar_planet(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from prior distributions
    rps = sample_rp(np.random.rand(N), np.full_like(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    Ptra = (rps*Rearth + R_s*Rsun)/a * e_corr

    # find instances with collisions
    coll = ((rps*Rearth + R_s*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False) & (qs_comp != 0.0)
        # calculate lnL for transiting systems
        a_arr = np.full(N, a)
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a_arr[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[mask],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (coll[i] == False) 
                & (qs_comp[i] != 0.0)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a, R_s, u1, u2,
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': rps[idx],
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': np.zeros(N_samples),
        'R_EB': np.zeros(N_samples),
        'fluxratio_EB': np.zeros(N_samples),
        'fluxratio_comp': fluxratios_comp[idx],
        'lnZ': lnZ
    }
    return res


def lnZ_PEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
    """
    Calculates the marginal likelihood of the PEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # sample from prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))
    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1/M_s] = 0.1/M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    # calculate properties of the drawn EBs
    masses = qs*M_s
    radii, Teffs = stellar_relations(
        masses, np.full(N, R_s), np.full(N, Teff)
        )
    # calculate flux ratios in the TESS band
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(np.array([M_s])))
        )

    # calculate properties of the drawn companions
    masses_comp = qs_comp*M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
        )
    # calculate flux ratios in the TESS band
    fluxratios_comp = (
        flux_relation(masses_comp)
        / (flux_relation(masses_comp) + flux_relation(np.array([M_s])))
        )

    # calculate priors for companions
    if molusc_file is None:
        if contrast_curve_file is None:
            # use TESS/Vis band flux ratios
            delta_mags = 2.5*np.log10(
                fluxratios_comp/(1-fluxratios_comp)
                )
            lnprior_companion = lnprior_bound_EB(
                M_s, plx, np.abs(delta_mags),
                np.array([2.2]), np.array([1.0])
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # use flux ratio of contrast curve filter
            fluxratios_comp_cc = (
                flux_relation(masses_comp, filt)
                / (flux_relation(masses_comp, filt)
                    + flux_relation(np.array([M_s]), filt))
                )
            delta_mags = 2.5*np.log10(
                fluxratios_comp_cc/(1-fluxratios_comp_cc)
                )
            separations, contrasts = file_to_contrast_curve(
                contrast_curve_file
                )
            lnprior_companion = lnprior_bound_EB(
                M_s, plx, np.abs(delta_mags), separations, contrasts
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    # calculate short-period binary prior for star of mass M_s
    lnprior_Mstar = lnprior_Mstar_binary(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
    a_twin = ((G*(M_s+masses)*Msun)/(4*pi**2)*(2*P_orb*86400)**2)**(1/3)
    Ptra_twin = (radii*Rsun + R_s*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(R_s*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + R_s*Rsun) > a*(1-eccs))
    coll_twin = ((2*R_s*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = ((incs >= inc_min) & (coll == False)
            & (qs < 0.95) & (qs_comp != 0.0))
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[mask],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = ((incs >= inc_min) & (coll_twin == False) 
            & (qs >= 0.95) & (qs_comp != 0.0))
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[mask],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] < 0.95) 
                & (coll[i] == False) & (qs_comp[i] != 0)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95) 
                & (coll_twin[i] == False) & (qs_comp[i] != 0)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, radii[i], fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': fluxratios_comp[idx],
        'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL_twin+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res_twin = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, 2*P_orb),
        'inc': incs[idx],
        'b': b_twin[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': fluxratios_comp[idx],
        'lnZ': lnZ
    }
    return res, res_twin


def lnZ_STP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float, Z: float,
            plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
    """
    Calculates the marginal likelihood of the STP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)

    # sample from q prior distribution
    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1/M_s] = 0.1/M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    # calculate properties of the drawn companions
    masses_comp = qs_comp*M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
        )
    loggs_comp = np.log10(G*(masses_comp*Msun)/(radii_comp*Rsun)**2)
    # calculate flux ratios in the TESS band
    fluxratios_comp = (
        flux_relation(masses_comp)
        / (flux_relation(masses_comp) + flux_relation(np.array([M_s])))
        )

    # calculate limb darkening ceofficients for companions
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
        ldc_at_Z = ldc_T[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
        ldc_at_Z = ldc_K[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
    rounded_loggs_comp = np.round(loggs_comp/0.5) * 0.5
    rounded_loggs_comp[rounded_loggs_comp < 3.5] = 3.5
    rounded_loggs_comp[rounded_loggs_comp > 5.0] = 5.0
    rounded_Teffs_comp = np.round(Teffs_comp/250) * 250
    rounded_Teffs_comp[rounded_Teffs_comp < 3500] = 3500
    rounded_Teffs_comp[rounded_Teffs_comp > 10000] = 10000
    u1s_comp, u2s_comp = np.zeros(N), np.zeros(N)
    for i, (comp_Teff, comp_logg) in enumerate(
            zip(rounded_Teffs_comp, rounded_loggs_comp)
            ):
        mask = (Teffs_at_Z == comp_Teff) & (loggs_at_Z == comp_logg)
        u1s_comp[i], u2s_comp[i] = u1s_at_Z[mask], u2s_at_Z[mask]

    # calculate priors for companions
    if molusc_file is None:
        if contrast_curve_file is None:
            # use TESS/Vis band flux ratios
            delta_mags = 2.5*np.log10(fluxratios_comp/(1-fluxratios_comp))
            lnprior_companion = lnprior_bound_TP(
                M_s, plx, np.abs(delta_mags),
                np.array([2.2]), np.array([1.0])
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # use flux ratio of contrast curve filter
            fluxratios_comp_cc = (
                flux_relation(masses_comp, filt)
                / (flux_relation(masses_comp, filt)
                    + flux_relation(np.array([M_s]), filt))
                )
            delta_mags = 2.5*np.log10(fluxratios_comp_cc/(1-fluxratios_comp_cc))
            separations, contrasts = file_to_contrast_curve(
                contrast_curve_file
                )
            lnprior_companion = lnprior_bound_TP(
                M_s, plx, np.abs(delta_mags), separations, contrasts
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    # calculate short-period planet prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_planet(masses_comp)

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from prior distributions
    rps = sample_rp(np.random.rand(N), masses_comp, flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*masses_comp*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (rps*Rearth + radii_comp*Rsun)/a * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(radii_comp*Rsun)

    # find instances with collisions
    coll = ((rps*Rearth + radii_comp*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False) & (qs_comp != 0.0)
        # calculate lnL for transiting systems
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a[mask], radii_comp[mask],
                    u1s_comp[mask], u2s_comp[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[mask],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (coll[i] == False)
                & (qs_comp[i] != 0.0)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a[i], radii_comp[i],
                    u1s_comp[i], u2s_comp[i],
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
     'M_s': masses_comp[idx],
     'R_s': radii_comp[idx],
     'u1': u1s_comp[idx],
     'u2': u2s_comp[idx],
     'P_orb': np.full(N_samples, P_orb),
     'inc': incs[idx],
     'b': b[idx],
     'R_p': rps[idx],
     'ecc': eccs[idx],
     'argp': argps[idx],
     'M_EB': np.zeros(N_samples),
     'R_EB': np.zeros(N_samples),
     'fluxratio_EB': np.zeros(N_samples),
     'fluxratio_comp': fluxratios_comp[idx],
     'lnZ': lnZ
    }
    return res


def lnZ_SEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
    """
    Calculates the marginal likelihood of the SEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)

    # sample from prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))
    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1/M_s] = 0.1/M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))   

    # calculate properties of the drawn companions
    masses_comp = qs_comp*M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
        )
    loggs_comp = np.log10(G*(masses_comp*Msun)/(radii_comp*Rsun)**2)
    # calculate flux ratios in the TESS band
    fluxratios_comp = (
        flux_relation(masses_comp)
        / (flux_relation(masses_comp) + flux_relation(np.array([M_s])))
        )

    # calculate limb darkening ceofficients for companions
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
        ldc_at_Z = ldc_T[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
        ldc_at_Z = ldc_K[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
    rounded_loggs_comp = np.round(loggs_comp/0.5) * 0.5
    rounded_loggs_comp[rounded_loggs_comp < 3.5] = 3.5
    rounded_loggs_comp[rounded_loggs_comp > 5.0] = 5.0
    rounded_Teffs_comp = np.round(Teffs_comp/250) * 250
    rounded_Teffs_comp[rounded_Teffs_comp < 3500] = 3500
    rounded_Teffs_comp[rounded_Teffs_comp > 13000] = 13000
    u1s_comp, u2s_comp = np.zeros(N), np.zeros(N)
    for i, (comp_Teff, comp_logg) in enumerate(
            zip(rounded_Teffs_comp, rounded_loggs_comp)
            ):
        mask = (Teffs_at_Z == comp_Teff) & (loggs_at_Z == comp_logg)
        u1s_comp[i], u2s_comp[i] = u1s_at_Z[mask], u2s_at_Z[mask]

    # calculate properties of the drawn EBs
    masses = qs*masses_comp
    radii, Teffs = stellar_relations(masses, radii_comp, Teffs_comp)
    # calculate flux ratios in the TESS band
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(np.array([M_s])))
        )

    # calculate priors for companions
    if molusc_file is None:
        if contrast_curve_file is None:
            # use TESS/Vis band flux ratios
            delta_mags = 2.5*np.log10(
                (fluxratios_comp/(1-fluxratios_comp))
                + (fluxratios/(1-fluxratios))
                )
            lnprior_companion = lnprior_bound_EB(
                M_s, plx, np.abs(delta_mags),
                np.array([2.2]), np.array([1.0])
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # use flux ratio of contrast curve filter
            fluxratios_cc = (
                flux_relation(masses, filt)
                / (flux_relation(masses, filt)
                    + flux_relation(np.array([M_s]), filt))
                )
            fluxratios_comp_cc = (
                flux_relation(masses_comp, filt)
                / (flux_relation(masses_comp, filt)
                    + flux_relation(np.array([M_s]), filt))
                )
            delta_mags = 2.5*np.log10(
                (fluxratios_comp_cc/(1-fluxratios_comp_cc))
                + (fluxratios_cc/(1-fluxratios_cc))
                )
            separations, contrasts = file_to_contrast_curve(
                contrast_curve_file
                )
            lnprior_companion = lnprior_bound_EB(
                M_s, plx, np.abs(delta_mags), separations, contrasts
                )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    # calculate short-period binary prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_binary(masses_comp)

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = (
        (G*(masses_comp+masses)*Msun)/(4*pi**2)*(P_orb*86400)**2
        )**(1/3)
    Ptra = (radii*Rsun + radii_comp*Rsun)/a * e_corr
    a_twin = (
        (G*(masses_comp+masses)*Msun)/(4*pi**2)*(2*P_orb*86400)**2
        )**(1/3)
    Ptra_twin = (radii*Rsun + radii_comp*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(radii_comp*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(radii_comp*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + radii_comp*Rsun) > a*(1-eccs))
    coll_twin = ((2*radii_comp*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = ((incs >= inc_min) & (coll == False) 
            & (qs < 0.95) & (qs_comp != 0.0))
        # calculate lnL for transiting systems
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask], radii_comp[mask],
                    u1s_comp[mask], u2s_comp[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[mask],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = ((incs >= inc_min) & (coll_twin == False) 
            & (qs >= 0.95) & (qs_comp != 0.0))
        # calculate lnL for transiting systems
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask], radii_comp[mask],
                    u1s_comp[mask], u2s_comp[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[mask],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] < 0.95) 
                & (coll[i] == False) & (qs_comp[i] != 0.0)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], radii_comp[i],
                    u1s_comp[i], u2s_comp[i],
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95)
                & (coll_twin[i] == False) & (qs_comp[i] != 0.0)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, radii[i], fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], radii_comp[i],
                    u1s_comp[i], u2s_comp[i],
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL + lnprior_companion + lnprior_Mstar + lnprior_Porb + 600)
            )
        )
    lnZ = np.log(Z)
    res = {
     'M_s': masses_comp[idx],
     'R_s': radii_comp[idx],
     'u1': u1s_comp[idx],
     'u2': u2s_comp[idx],
     'P_orb': np.full(N_samples, P_orb),
     'inc': incs[idx],
     'b': b[idx],
     'R_p': np.zeros(N_samples),
     'ecc': eccs[idx],
     'argp': argps[idx],
     'M_EB': masses[idx],
     'R_EB': radii[idx],
     'fluxratio_EB': fluxratios[idx],
     'fluxratio_comp': fluxratios_comp[idx],
     'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL_twin+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res_twin = {
     'M_s': masses_comp[idx],
     'R_s': radii_comp[idx],
     'u1': u1s_comp[idx],
     'u2': u2s_comp[idx],
     'P_orb': np.full(N_samples, 2*P_orb),
     'inc': incs[idx],
     'b': b_twin[idx],
     'R_p': np.zeros(N_samples),
     'ecc': eccs[idx],
     'argp': argps[idx],
     'M_EB': masses[idx],
     'R_EB': radii[idx],
     'fluxratio_EB': fluxratios[idx],
     'fluxratio_comp': fluxratios_comp[idx],
     'lnZ': lnZ
    }
    return res, res_twin


def lnZ_DTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, output_url: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the DTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        output_url (string): Link to trilegal query results.
        contrast_curve_file (string): Contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    a = ((G*M_s*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # determine background star population properties
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = (
        trilegal_results(output_url, Tmag)
        )
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10**(delta_mags/2.5) / (1 + 10**(delta_mags/2.5))
    N_comp = Tmags_comp.shape[0]
    # draw random sample of background stars
    idxs = np.random.randint(0, N_comp-1, N)

    # calculate priors for companions
    if contrast_curve_file is None:
        # use TESS/Vis band flux ratios
        delta_mags = 2.5*np.log10(
            fluxratios_comp[idxs]/(1-fluxratios_comp[idxs])
            )
        lnprior_companion = np.full(
            N, np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        if filt == "J":
            delta_mags = delta_Jmags[idxs]
        elif filt == "H":
            delta_mags = delta_Hmags[idxs]
        elif filt == "K":
            delta_mags = delta_Kmags[idxs]
        else:
            delta_mags = delta_mags[idxs]
        separations, contrasts = file_to_contrast_curve(
            contrast_curve_file
            )
        lnprior_companion = lnprior_background(
            N_comp, np.abs(delta_mags), separations, contrasts
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf

    # calculate short-period planet prior for star of mass M_s
    lnprior_Mstar = lnprior_Mstar_planet(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from R_p and inc prior distributions
    rps = sample_rp(np.random.rand(N), np.full_like(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    Ptra = (rps*Rearth + R_s*Rsun)/a * e_corr

    # find instances with collisions
    coll = ((rps*Rearth + R_s*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False)
        # calculate lnL for transiting systems
        a_arr = np.full(N, a)
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a_arr[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[idxs[mask]],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a, R_s, u1, u2,
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': rps[idx],
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': np.zeros(N_samples),
        'R_EB': np.zeros(N_samples),
        'fluxratio_EB': np.zeros(N_samples),
        'fluxratio_comp': fluxratios_comp[idxs[idx]],
        'lnZ': lnZ
    }
    return res


def lnZ_DEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, output_url: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the DEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        output_url (string): Link to trilegal query results.
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # sample from inc and q prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate properties of the drawn EBs
    masses = qs*M_s
    radii, Teffs = stellar_relations(
        masses, np.full(N, R_s), np.full(N, Teff)
        )
    # calculate flux ratios in the TESS band
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(np.array([M_s])))
        )

    # determine background star population properties
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = (
        trilegal_results(output_url, Tmag)
        )
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10**(delta_mags/2.5) / (1 + 10**(delta_mags/2.5))
    N_comp = Tmags_comp.shape[0]
    # draw random sample of background stars
    idxs = np.random.randint(0, N_comp-1, N)

    # calculate priors for companions
    if contrast_curve_file is None:
        # use TESS/Vis band flux ratios
        delta_mags = 2.5*np.log10(
            fluxratios_comp[idxs]/(1-fluxratios_comp[idxs])
            )
        lnprior_companion = np.full(
            N, np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        if filt == "J":
            delta_mags = delta_Jmags[idxs]
        elif filt == "H":
            delta_mags = delta_Hmags[idxs]
        elif filt == "K":
            delta_mags = delta_Kmags[idxs]
        else:
            delta_mags = delta_mags[idxs]
        separations, contrasts = file_to_contrast_curve(
            contrast_curve_file
            )
        lnprior_companion = lnprior_background(
            N_comp, np.abs(delta_mags), separations, contrasts
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf

    # calculate short-period binary prior for star of mass M_s
    lnprior_Mstar = lnprior_Mstar_binary(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
    a_twin = ((G*(M_s+masses)*Msun)/(4*pi**2)*(2*P_orb*86400)**2)**(1/3)
    Ptra_twin = (radii*Rsun + R_s*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(R_s*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + R_s*Rsun) > a*(1-eccs))
    coll_twin = ((2*R_s*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95)
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[idxs[mask]],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95)
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[idxs[mask]],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] < 0.95) & (coll[i] == False):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95)
                & (coll_twin[i] == False)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, radii[i], fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=False,
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': fluxratios_comp[idxs[idx]],
        'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL_twin+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res_twin = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, 2*P_orb),
        'inc': incs[idx],
        'b': b_twin[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': fluxratios_comp[idxs[idx]],
        'lnZ': lnZ
    }
    return res, res_twin


def lnZ_BTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag: float, Hmag: float, Kmag: float,
            output_url: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the BTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        output_url (string): Link to trilegal query results.
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)

    # determine background star population properties
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = (
        trilegal_results(output_url, Tmag)
        )
    radii_comp = np.sqrt(G*masses_comp*Msun / 10**loggs_comp) / Rsun
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10**(delta_mags/2.5) / (1 + 10**(delta_mags/2.5))
    N_comp = Tmags_comp.shape[0]
    # determine limb darkening coefficients of background stars
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_comp, u2s_comp = np.zeros(N_comp), np.zeros(N_comp)
    for i in range(N_comp):
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teffs_comp[i]))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-loggs_comp[i]))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Zs_comp[i]))]
        mask = (
            (ldc_Zs == this_Z)
            & (ldc_Teffs == this_Teff)
            & (ldc_loggs == this_logg)
            )
        u1s_comp[i], u2s_comp[i] = ldc_u1s[mask], ldc_u2s[mask]
    # draw random sample of background stars
    idxs = np.random.randint(0, N_comp, N)

    # calculate priors for companions
    if contrast_curve_file is None:
        # use TESS/Vis band flux ratios
        delta_mags = 2.5*np.log10(
            fluxratios_comp[idxs]/(1-fluxratios_comp[idxs])
            )
        lnprior_companion = np.full(
            N, np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        if filt == "J":
            delta_mags = delta_Jmags[idxs]
        elif filt == "H":
            delta_mags = delta_Hmags[idxs]
        elif filt == "K":
            delta_mags = delta_Kmags[idxs]
        else:
            delta_mags = delta_mags[idxs]
        separations, contrasts = file_to_contrast_curve(
            contrast_curve_file
            )
        lnprior_companion = lnprior_background(
            N_comp, np.abs(delta_mags), separations, contrasts
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf

    # calculate short-prior planet prior for stars of masses masses_comp
    lnprior_Mstar = lnprior_Mstar_planet(masses_comp[idxs])

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from inc and R_p prior distributions
    rps = sample_rp(np.random.rand(N), masses_comp[idxs], flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*masses_comp[idxs]*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (rps*Rearth + radii_comp[idxs]*Rsun)/a * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(radii_comp[idxs]*Rsun)

    # find instances with collisions
    coll = ((rps*Rearth + radii_comp[idxs]*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (loggs_comp[idxs] >= 3.5)
            & (Teffs_comp[idxs] <= 10000)
            )
        # calculate lnL for transiting systems
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a[mask], radii_comp[idxs[mask]],
                    u1s_comp[idxs[mask]], u2s_comp[idxs[mask]],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[idxs[mask]],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (loggs_comp[idxs[i]] >= 3.5)
                    & (Teffs_comp[idxs[i]] <= 10000) & (coll[i] == False)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a[i], radii_comp[idxs[i]],
                    u1s_comp[idxs[i]], u2s_comp[idxs[i]],
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
        'M_s': masses_comp[idxs[idx]],
        'R_s': radii_comp[idxs[idx]],
        'u1': u1s_comp[idxs[idx]],
        'u2': u2s_comp[idxs[idx]],
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': rps[idx],
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': np.zeros(N_samples),
        'R_EB': np.zeros(N_samples),
        'fluxratio_EB': np.zeros(N_samples),
        'fluxratio_comp': fluxratios_comp[idxs[idx]],
        'lnZ': lnZ
    }
    return res


def lnZ_BEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag:float, Hmag: float, Kmag: float,
            output_url: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 1000000, parallel: bool = False,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the BEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        output_url (string): Link to trilegal query results.
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)

    # sample from prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    qs_comp = sample_q_companion(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # determine background star population properties
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = (
        trilegal_results(output_url, Tmag)
        )
    radii_comp = np.sqrt(G*masses_comp*Msun / 10**loggs_comp) / Rsun
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10**(delta_mags/2.5) / (1 + 10**(delta_mags/2.5))
    fluxratios_comp_J = 10**(delta_Jmags/2.5) / (1 + 10**(delta_Jmags/2.5))
    fluxratios_comp_H = 10**(delta_Hmags/2.5) / (1 + 10**(delta_Hmags/2.5))
    fluxratios_comp_K = 10**(delta_Kmags/2.5) / (1 + 10**(delta_Kmags/2.5))
    N_comp = Tmags_comp.shape[0]
    # determine limb darkening coefficients of background stars
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_comp, u2s_comp = np.zeros(N_comp), np.zeros(N_comp)
    for i in range(N_comp):
        this_Teff = ldc_Teffs[np.argmin(
            np.abs(ldc_Teffs-Teffs_comp[i])
            )]
        this_logg = ldc_loggs[np.argmin(
            np.abs(ldc_loggs-loggs_comp[i])
            )]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Zs_comp[i]))]
        mask = (
            (ldc_Zs == this_Z)
            & (ldc_Teffs == this_Teff)
            & (ldc_loggs == this_logg)
            )
        u1s_comp[i], u2s_comp[i] = ldc_u1s[mask], ldc_u2s[mask]
    # draw random sample of background stars
    idxs = np.random.randint(0, N_comp, N)

    # calculate properties of the drawn EBs
    masses = qs*masses_comp[idxs]
    radii, Teffs = stellar_relations(
        masses, radii_comp[idxs], Teffs_comp[idxs]
        )
    # calculate EB flux ratios in the TESS band
    fluxratios_comp_bound = (
        flux_relation(masses_comp[idxs])
        / (
            flux_relation(masses_comp[idxs])
            + flux_relation(np.array([M_s]))
            )
        )
    distance_correction = fluxratios_comp[idxs]/fluxratios_comp_bound
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(np.array([M_s])))
        * distance_correction
        )
    # calculate EB flux ratios in the contrast curve filter
    if filt == "J":
        fluxratios_comp_cc = fluxratios_comp_J[idxs]
    elif filt == "H":
        fluxratios_comp_cc = fluxratios_comp_H[idxs]
    elif filt == "K":
        fluxratios_comp_cc = fluxratios_comp_K[idxs]
    else:
        fluxratios_comp_cc = fluxratios_comp[idxs]
    fluxratios_comp_bound_cc = (
        flux_relation(masses_comp[idxs], filt)
        / (
            flux_relation(masses_comp[idxs], filt)
            + flux_relation(np.array([M_s]), filt)
            )
        )
    distance_correction_cc = fluxratios_comp_cc/fluxratios_comp_bound_cc
    fluxratios_cc = (
        flux_relation(masses, filt)
        / (flux_relation(masses, filt)
            + flux_relation(np.array([M_s]), filt))
        * distance_correction_cc
        )

    # calculate priors for companions
    if contrast_curve_file is None:
        # use TESS/Vis band flux ratios
        delta_mags = 2.5*np.log10(
            (fluxratios_comp[idxs]/(1-fluxratios_comp[idxs]))
            + (fluxratios/(1-fluxratios))
            )
        lnprior_companion = np.full(
            N, np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        # use contrast curve filter flux ratios
        delta_mags = 2.5*np.log10(
            (fluxratios_comp_cc/(1-fluxratios_comp_cc))
            + (fluxratios_cc/(1-fluxratios_cc))
            )
        separations, contrasts = file_to_contrast_curve(
            contrast_curve_file
            )
        lnprior_companion = lnprior_background(
            N_comp, np.abs(delta_mags), separations, contrasts
            )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf

    # calculate short-period binary prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_binary(masses_comp[idxs])

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = (
        (G*(masses_comp[idxs]+masses)*Msun)/(4*pi**2)*(P_orb*86400)**2
        )**(1/3)
    Ptra = (radii*Rsun + radii_comp[idxs]*Rsun)/a * e_corr
    a_twin = (
        (G*(masses_comp[idxs]+masses)*Msun)/(4*pi**2)*(2*P_orb*86400)**2
        )**(1/3)
    Ptra_twin = (radii*Rsun + radii_comp[idxs]*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(radii_comp[idxs]*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(radii_comp[idxs]*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + radii_comp[idxs]*Rsun) > a*(1-eccs))
    coll_twin = ((2*radii_comp[idxs]*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (
            (incs >= inc_min)
            & (coll_twin == False)
            & (qs < 0.95)
            & (loggs_comp[idxs] >= 3.5)
            & (Teffs_comp[idxs] <= 10000)
            )
        # calculate lnL for transiting systems
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask], radii_comp[idxs[mask]],
                    u1s_comp[idxs[mask]], u2s_comp[idxs[mask]],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[idxs[mask]],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (
            (incs >= inc_min)
            & (coll_twin == False)
            & (qs >= 0.95)
            & (loggs_comp[idxs] >= 3.5)
            & (Teffs_comp[idxs] <= 10000)
            )
        # calculate lnL for transiting systems
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask], radii_comp[idxs[mask]],
                    u1s_comp[idxs[mask]], u2s_comp[idxs[mask]],
                    eccs[mask], argps[mask],
                    companion_fluxratio=fluxratios_comp[idxs[mask]],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] < 0.95)
                    & (loggs_comp[idxs[i]] >= 3.5)
                    & (Teffs_comp[idxs[i]] <= 10000)
                    & (coll[i] == False)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], radii_comp[idxs[i]],
                    u1s_comp[idxs[i]], u2s_comp[idxs[i]],
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95)
                    & (loggs_comp[idxs[i]] >= 3.5)
                    & (Teffs_comp[idxs[i]] <= 10000)
                    & (coll_twin[i] == False)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, radii[i], fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], radii_comp[idxs[i]],
                    u1s_comp[idxs[i]], u2s_comp[idxs[i]],
                    eccs[i], argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=True,
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res = {
        'M_s': masses_comp[idxs[idx]],
        'R_s': radii_comp[idxs[idx]],
        'u1': u1s_comp[idxs[idx]],
        'u2': u2s_comp[idxs[idx]],
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': fluxratios_comp[idxs[idx]],
        'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(
        np.nan_to_num(
            np.exp(lnL_twin+lnprior_companion+lnprior_Mstar+lnprior_Porb+600)
            )
        )
    lnZ = np.log(Z)
    res_twin = {
        'M_s': masses_comp[idxs[idx]],
        'R_s': radii_comp[idxs[idx]],
        'u1': u1s_comp[idxs[idx]],
        'u2': u2s_comp[idxs[idx]],
        'P_orb': np.full(N_samples, 2*P_orb),
        'inc': incs[idx],
        'b': b_twin[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': fluxratios_comp[idxs[idx]],
        'lnZ': lnZ
    }
    return res, res_twin


def lnZ_NTP_unknown(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, Tmag: float, output_url: str,
                    N: int = 1000000, parallel: bool = False,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NTP scenario for
    a star of unknown properties.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        Tmag (float): Target star TESS magnitude.
        output_url (string): Link to trilegal query results.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)

    # determine properties of possible stars
    (Tmags_nearby, masses_nearby, loggs_nearby, Teffs_nearby,
        Zs_nearby, Jmags_nearby, Hmags_nearby, Kmags_nearby) = (
        trilegal_results(output_url, Tmag)
        )
    mask = (Tmag-1 < Tmags_nearby) & (Tmags_nearby < Tmag+1)
    Tmags_possible = Tmags_nearby[mask]
    masses_possible = masses_nearby[mask]
    loggs_possible = loggs_nearby[mask]
    Teffs_possible = Teffs_nearby[mask]
    Zs_possible = Zs_nearby[mask]
    radii_possible = np.sqrt(
        G*masses_possible*Msun / 10**loggs_possible
        ) / Rsun
    N_possible = Tmags_possible.shape[0]
    # determine limb darkening coefficients of background stars
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_possible = np.zeros(N_possible)
    u2s_possible = np.zeros(N_possible)
    for i in range(N_possible):
        this_Teff = ldc_Teffs[np.argmin(
            np.abs(ldc_Teffs-Teffs_possible[i])
            )]
        this_logg = ldc_loggs[np.argmin(
            np.abs(ldc_loggs-loggs_possible[i])
            )]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Zs_possible[i]))]
        mask = (
            (ldc_Zs == this_Z)
            & (ldc_Teffs == this_Teff)
            & (ldc_loggs == this_logg)
            )
        u1s_possible[i], u2s_possible[i] = ldc_u1s[mask], ldc_u2s[mask]
    # draw random sample of background stars
    if N_possible > 0:
        idxs = np.random.randint(0, N_possible, N)
    # if there aren't enough similar stars, don't do the calculation
    else:
        res = {
            'M_s': 0,
            'R_s': 0,
            'u1': 0,
            'u2': 0,
            'P_orb': P_orb,
            'inc': 0,
            'R_p': 0,
            'ecc': 0,
            'argp': 0,
            'M_EB': 0,
            'R_EB': 0,
            'fluxratio_EB': 0,
            'fluxratio_comp': 0,
            'lnZ': -np.inf
        }
        return res

    # calculate short-prior planet prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_planet(masses_possible[idxs])

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from inc and R_p prior distributions
    rps = sample_rp(np.random.rand(N), masses_possible[idxs], flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = (
        (G*masses_possible[idxs]*Msun)/(4*pi**2)*(P_orb*86400)**2
        )**(1/3)
    Ptra = (rps*Rearth + radii_possible[idxs]*Rsun)/a * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(radii_possible[idxs]*Rsun)

    # find instances with collisions
    coll = ((rps*Rearth + radii_possible[idxs]*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (loggs_possible[idxs] >= 3.5)
            & (Teffs_possible[idxs] <= 10000)
            )
        # calculate lnL for transiting systems
        companion_fluxratio = np.zeros(N)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a[mask],
                    radii_possible[idxs[mask]],
                    u1s_possible[idxs[mask]],
                    u2s_possible[idxs[mask]],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (loggs_possible[idxs[i]] >= 3.5)
                    & (Teffs_possible[idxs[i]] <= 10000)
                    & (coll[i] == False)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a[i], radii_possible[idxs[i]],
                    u1s_possible[idxs[i]], u2s_possible[idxs[i]],
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL+lnprior_Mstar+lnprior_Porb+600)
        ))
    lnZ = np.log(Z)
    res = {
        'M_s': masses_possible[idxs[idx]],
        'R_s': radii_possible[idxs[idx]],
        'u1': u1s_possible[idxs[idx]],
        'u2': u2s_possible[idxs[idx]],
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': rps[idx],
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': np.zeros(N_samples),
        'R_EB': np.zeros(N_samples),
        'fluxratio_EB': np.zeros(N_samples),
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    return res


def lnZ_NEB_unknown(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, Tmag: float, output_url: str,
                    N: int = 1000000, parallel: bool = False,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NEB scenario for a star
    of unknown properties.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        Tmag (float): Target star TESS magnitude.
        output_url (string): Link to trilegal query results.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)

    # sample from prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), 1.0)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # determine properties of possible stars
    (Tmags_nearby, masses_nearby, loggs_nearby, Teffs_nearby,
        Zs_nearby, Jmags_nearby, Hmags_nearby, Kmags_nearby) = (
        trilegal_results(output_url, Tmag)
        )
    mask = (Tmag-1 < Tmags_nearby) & (Tmags_nearby < Tmag+1)
    Tmags_possible = Tmags_nearby[mask]
    masses_possible = masses_nearby[mask]
    loggs_possible = loggs_nearby[mask]
    Teffs_possible = Teffs_nearby[mask]
    Zs_possible = Zs_nearby[mask]
    radii_possible = np.sqrt(
        G*masses_possible*Msun / 10**loggs_possible
        ) / Rsun
    N_possible = Tmags_possible.shape[0]
    # determine limb darkening coefficients of background stars
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_possible = np.zeros(N_possible)
    u2s_possible = np.zeros(N_possible)
    for i in range(N_possible):
        this_Teff = ldc_Teffs[np.argmin(
            np.abs(ldc_Teffs-Teffs_possible[i])
            )]
        this_logg = ldc_loggs[np.argmin(
            np.abs(ldc_loggs-loggs_possible[i])
            )]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Zs_possible[i]))]
        mask = (
            (ldc_Zs == this_Z)
            & (ldc_Teffs == this_Teff)
            & (ldc_loggs == this_logg)
            )
        u1s_possible[i], u2s_possible[i] = ldc_u1s[mask], ldc_u2s[mask]
    # draw random sample of background stars
    if N_possible > 0:
        idxs = np.random.randint(0, N_possible, N)
    # if there aren't enough similar stars, don't do the calculation
    else:
        res = {
            'M_s': 0,
            'R_s': 0,
            'u1': 0,
            'u2': 0,
            'P_orb': P_orb,
            'inc': 0,
            'b': 0,
            'R_p': 0,
            'ecc': 0,
            'argp': 0,
            'M_EB': 0,
            'R_EB': 0,
            'fluxratio_EB': 0,
            'fluxratio_comp': 0,
            'lnZ': -np.inf
        }
        return res

    # calculate properties of the drawn EBs
    masses = qs*masses_possible[idxs]
    radii, Teffs = stellar_relations(
        masses, radii_possible[idxs], Teffs_possible[idxs]
        )
    # calculate flux ratios in the TESS band
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(masses_possible[idxs]))
        )

    # calculate short-period binary prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_binary(masses_possible[idxs])

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = (
        (G*(masses_possible[idxs]+masses)*Msun)/(4*pi**2)
        * (P_orb*86400)**2
        )**(1/3)
    Ptra = (radii*Rsun + radii_possible[idxs]*Rsun)/a * e_corr
    a_twin = (
        (G*(masses_possible[idxs]+masses)*Msun)/(4*pi**2)
        * (2*P_orb*86400)**2
        )**(1/3)
    Ptra_twin = (radii*Rsun + radii_possible[idxs]*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(radii_possible[idxs]*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(radii_possible[idxs]*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + radii_possible[idxs]*Rsun) > a*(1-eccs))
    coll_twin = ((2*radii_possible[idxs]*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (qs < 0.95)
            & (loggs_possible[idxs] >= 3.5)
            & (Teffs_possible[idxs] <= 10000)
            )
        # calculate lnL for transiting systems
        companion_fluxratio = np.zeros(N)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask],
                    radii_possible[idxs[mask]],
                    u1s_possible[idxs[mask]], u2s_possible[idxs[mask]],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (qs >= 0.95)
            & (loggs_possible[idxs] >= 3.5)
            & (Teffs_possible[idxs] <= 10000)
            )
        # calculate lnL for transiting systems
        companion_fluxratio = np.zeros(N)
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask],
                    radii_possible[idxs[mask]],
                    u1s_possible[idxs[mask]], u2s_possible[idxs[mask]],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] < 0.95)
                    & (loggs_possible[idxs[i]] >= 3.5)
                    & (Teffs_possible[idxs[i]] <= 10000)
                    & (coll[i] == False)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], radii_possible[idxs[i]],
                    u1s_possible[idxs[i]], u2s_possible[idxs[i]],
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95)
                    & (loggs_possible[idxs[i]] >= 3.5)
                    & (Teffs_possible[idxs[i]] <= 10000)
                    & (coll_twin[i] == False)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, radii[i], fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], radii_possible[idxs[i]],
                    u1s_possible[idxs[i]], u2s_possible[idxs[i]],
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL+lnprior_Mstar+lnprior_Porb+600)
        ))
    lnZ = np.log(Z)
    res = {
        'M_s': masses_possible[idxs[idx]],
        'R_s': radii_possible[idxs[idx]],
        'u1': u1s_possible[idxs[idx]],
        'u2': u2s_possible[idxs[idx]],
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL_twin+lnprior_Mstar+lnprior_Porb+600)
        ))
    lnZ = np.log(Z)
    res_twin = {
        'M_s': masses_possible[idxs[idx]],
        'R_s': radii_possible[idxs[idx]],
        'u1': u1s_possible[idxs[idx]],
        'u2': u2s_possible[idxs[idx]],
        'P_orb': np.full(N_samples, 2*P_orb),
        'inc': incs[idx],
        'b': b_twin[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    return res, res_twin


def lnZ_NTP_evolved(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, R_s: float, Teff: float, Z: float,
                    N: int = 1000000, parallel: bool = False,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NTP scenario for
    subgiant stars.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    logg = 3.0
    M_s = (10**logg)*(R_s*Rsun)**2 / G / Msun
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # calculate short-prior planet prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_planet(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_planet(P_orb, flatpriors)

    # sample from inc and R_p prior distributions
    rps = sample_rp(np.random.rand(N), np.full_like(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*M_s*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (rps*Rearth + R_s*Rsun)/a * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)

    # find instances with collisions
    coll = ((rps*Rearth + R_s*Rsun) > a*(1-eccs))

    lnL = np.full(N, -np.inf)
    if parallel:
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False)
        # calculate lnL for transiting systems
        a_arr = np.full(N, a)
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_TP_p(
                    time, flux, sigma, rps[mask],
                    P_orb, incs[mask], a_arr[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_TP(
                    time, flux, sigma, rps[i],
                    P_orb, incs[i], a, R_s, u1, u2,
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )

    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL+lnprior_Mstar+lnprior_Porb+600)
        ))
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': rps[idx],
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': np.zeros(N_samples),
        'R_EB': np.zeros(N_samples),
        'fluxratio_EB': np.zeros(N_samples),
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    return res


def lnZ_NEB_evolved(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, R_s: float, Teff: float, Z: float,
                    N: int = 1000000, parallel: bool = False,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NEB scenario
    for subgiant stars.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    lnsigma = np.log(sigma)
    logg = 3.0
    M_s = (10**logg)*(R_s*Rsun)**2 / G / Msun
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # sample from inc and q prior distributions
    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), 1.0)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=P_orb)
    argps = sample_w(np.random.rand(N))

    # calculate properties of the drawn EBs
    masses = qs*M_s
    radii, Teffs = stellar_relations(
        masses, np.full(N, R_s), np.full(N, Teff)
        )
    fluxratios = (
        flux_relation(masses)
        / (flux_relation(masses) + flux_relation(np.array([M_s])))
        )

    # calculate short-period binary prior for stars
    # with masses masses_comp
    lnprior_Mstar = lnprior_Mstar_binary(np.array([M_s]))

    # calculate orbital period prior
    lnprior_Porb = lnprior_Porb_binary(P_orb)

    # calculate transit probability for each instance
    e_corr = (1+eccs*np.sin(argps*pi/180))/(1-eccs**2)
    a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
    Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
    a_twin = ((G*(M_s+masses)*Msun)/(4*pi**2)*(2*P_orb*86400)**2)**(1/3)
    Ptra_twin = (R_s*Rsun + R_s*Rsun)/a_twin * e_corr

    # calculate impact parameter
    r = a*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b = r*np.cos(incs*pi/180)/(R_s*Rsun)
    r_twin = a_twin*(1-eccs**2)/(1+eccs*np.sin(argps*np.pi/180)) 
    b_twin = r_twin*np.cos(incs*pi/180)/(R_s*Rsun)

    # find instances with collisions
    coll = ((radii*Rsun + R_s*Rsun) > a*(1-eccs))
    coll_twin = ((2*R_s*Rsun) > a_twin*(1-eccs))

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        # q < 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra <= 1.] = np.arccos(Ptra[Ptra <= 1.]) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95)
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = -0.5*ln2pi - lnsigma - lnL_EB_p(
                    time, flux, sigma, radii[mask], fluxratios[mask],
                    P_orb, incs[mask], a[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
        # q >= 0.95
        # find minimum inclination each planet can have while transiting
        inc_min = np.full(N, 90.)
        inc_min[Ptra_twin <= 1.] = np.arccos(
            Ptra_twin[Ptra_twin <= 1.]
            ) * 180./pi
        # filter out systems that do not transit or have a collision
        mask = (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95)
        # calculate lnL for transiting systems
        R_s_arr = np.full(N, R_s)
        u1_arr = np.full(N, u1)
        u2_arr = np.full(N, u2)
        companion_fluxratio = np.zeros(N)
        lnL_twin[mask] = -0.5*ln2pi - lnsigma - lnL_EB_twin_p(
                    time, flux, sigma, R_s, fluxratios[mask],
                    2*P_orb, incs[mask], a_twin[mask], R_s_arr[mask],
                    u1_arr[mask], u2_arr[mask],
                    eccs[mask], argps[mask],
                    companion_fluxratio=companion_fluxratio[mask],
                    exptime=exptime, nsamples=nsamples
                    )
    else:
        for i in range(N):
            # q < 0.95
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] < 0.95)
                    & (coll[i] == False)):
                lnL[i] = -0.5*ln2pi - lnsigma - lnL_EB(
                    time, flux, sigma, radii[i], fluxratios[i],
                    P_orb, incs[i], a[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )
            # q >= 0.95 and 2xP_orb
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180/pi
            else:
                continue
            if ((incs[i] >= inc_min) & (qs[i] >= 0.95)
                    & (coll_twin[i] == False)):
                lnL_twin[i] = -0.5*ln2pi - lnsigma - lnL_EB_twin(
                    time, flux, sigma, R_s, fluxratios[i],
                    2*P_orb, incs[i], a_twin[i], R_s, u1, u2,
                    eccs[i], argps[i],
                    exptime=exptime, nsamples=nsamples
                    )

    # results for q < 0.95
    N_samples = 100
    idx = (-lnL).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL + lnprior_Mstar + lnprior_Porb + 600)
        ))
    lnZ = np.log(Z)
    res = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, P_orb),
        'inc': incs[idx],
        'b': b[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': radii[idx],
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    # results for q >= 0.95 and 2xP_orb
    N_samples = 100
    idx = (-lnL_twin).argsort()[:N_samples]
    Z = np.mean(np.nan_to_num(
        np.exp(lnL_twin+lnprior_Mstar+lnprior_Porb+600)
        ))
    lnZ = np.log(Z)
    res_twin = {
        'M_s': np.full(N_samples, M_s),
        'R_s': np.full(N_samples, R_s),
        'u1': np.full(N_samples, u1),
        'u2': np.full(N_samples, u2),
        'P_orb': np.full(N_samples, 2*P_orb),
        'inc': incs[idx],
        'b': b_twin[idx],
        'R_p': np.zeros(N_samples),
        'ecc': eccs[idx],
        'argp': argps[idx],
        'M_EB': masses[idx],
        'R_EB': np.full(N_samples, R_s),
        'fluxratio_EB': fluxratios[idx],
        'fluxratio_comp': np.zeros(N_samples),
        'lnZ': lnZ
    }
    return res, res_twin
