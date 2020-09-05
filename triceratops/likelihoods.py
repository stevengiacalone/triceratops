import numpy as np
from astropy import constants
import batman

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi


def simulate_TP_transit(time: np.ndarray, R_p: float, P_orb: float,
                        inc: float, a: float, R_s: float, u1: float,
                        u2: float, companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False):
    """
    Simulates a transiting planet light curve using BATMAN.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_p (float): Planet radius [Earth radii].
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1-companion_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = P_orb
    params.rp = R_p*Rearth/(R_s*Rsun)
    params.a = a/(R_s*Rsun)
    params.inc = inc
    params.ecc = 0.0
    params.w = 0.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    flux = batman.TransitModel(params, time).light_curve(params)
    # step 2: adjust the light curve to account for flux dilution
    # from non-host star
    if companion_is_host:
        F_dilute = F_target / F_comp
        flux = (flux + F_dilute)/(1 + F_dilute)
    else:
        F_dilute = F_comp / F_target
        flux = (flux + F_dilute)/(1 + F_dilute)
    return flux


def simulate_EB_transit(time: np.ndarray, R_EB: float,
                        EB_fluxratio: float, P_orb: float, inc: float,
                        a: float, R_s: float, u1: float, u2: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False):
    """
    Simulates a eclipsing binary light curve using BATMAN.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        companion_fluxratio (float): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if it
                                  is not.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1-companion_fluxratio)
    F_EB = EB_fluxratio/(1-EB_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    params = batman.TransitParams()
    # calculate primary eclipse
    params.t0 = 0.0
    params.per = P_orb
    if R_EB/R_s == 1.0:
        params.rp = R_EB/R_s * 0.999
    else:
        params.rp = R_EB/R_s
    params.a = a/(R_s*Rsun)
    params.inc = inc
    params.ecc = 0.0
    params.w = 0.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    flux = batman.TransitModel(params, time).light_curve(params)
    # calculate secondary eclipse depth
    params.fp = (EB_fluxratio)/(1-EB_fluxratio)
    params.t_secondary = 0.5
    sec = batman.TransitModel(
        params, np.array([0.5, 0.75]), transittype="secondary"
        )
    sec_flux = sec.light_curve(params)[0]/max(sec.light_curve(params))
    # step 2: adjust the light curve to account for flux dilution
    # from EB and non-host star
    if companion_is_host:
        flux = (flux + F_EB/F_comp) / (1 + F_EB/F_comp)
        F_dilute = F_target / (F_comp + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    else:
        flux = (flux + F_EB/F_target) / (1 + F_EB/F_target)
        F_dilute = F_comp / (F_target + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    return flux, secdepth


def lnL_TP(time: np.ndarray, flux: np.ndarray, sigma: float, R_p: float,
           P_orb: float, inc: float, a: float, R_s: float,
           u1: float, u2: float, companion_fluxratio: float = 0.0,
           companion_is_host: bool = False):
    """
    Calculates the log likelihood of a transiting planet scenario by
    comparing a simulated light curve and the TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_p (float): Planet radius [Earth radii].
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
    Returns:
        Log likelihood (float).
    """
    model = simulate_TP_transit(
        time, R_p, P_orb, inc, a, R_s, u1, u2,
        companion_fluxratio, companion_is_host
        )
    return 0.5*(np.sum((flux-model)**2 / sigma**2))


def lnL_EB(time: np.ndarray, flux: np.ndarray, sigma: float,
           R_EB: float, EB_fluxratio: float, P_orb: float, inc: float,
           a: float, R_s: float, u1: float, u2: float,
           companion_fluxratio: float = 0.0,
           companion_is_host: bool = False):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q < 0.95 by comparing a simulated light curve and the
    TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
    Returns:
        Log likelihood (float).
    """
    model, secdepth = simulate_EB_transit(
        time, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        companion_fluxratio, companion_is_host
        )
    if secdepth < 1.5*sigma:
        return 0.5*(np.sum((flux-model)**2 / sigma**2))
    else:
        return np.inf


def lnL_EB_twin(time: np.ndarray, flux: np.ndarray, sigma: float,
                R_EB: float, EB_fluxratio: float, P_orb: float,
                inc: float, a: float, R_s: float, u1: float, u2: float,
                companion_fluxratio: float = 0.0,
                companion_is_host: bool = False):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q >= 0.95 and 2xP_orb by comparing a simulated light curve
    and the TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False
                                  if it is not.
    Returns:
        Log likelihood (float).
    """
    model, secdepth = simulate_EB_transit(
        time, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        companion_fluxratio, companion_is_host
        )
    return 0.5*(np.sum((flux-model)**2 / sigma**2))
