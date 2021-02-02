import numpy as np
from astropy import constants
from pytransit import QuadraticModel

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi

tm = QuadraticModel(interpolate=False)
tm_sec = QuadraticModel(interpolate=False)

def simulate_TP_transit(time: np.ndarray, R_p: float, P_orb: float,
                        inc: float, a: float, R_s: float, u1: float,
                        u2: float, ecc: float, argp: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False):
    """
    Simulates a transiting planet light curve using PyTransit.
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
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
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
    tm.set_data(time)
    flux = tm.evaluate_ps(
        k=R_p*Rearth/(R_s*Rsun),
        ldc=[float(u1), float(u2)],
        t0=0.0,
        p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=argp*(pi/180.)
        )
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
                        ecc: float, argp: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False):
    """
    Simulates an eclipsing binary light curve using PyTransit.
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
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if it
                                  is not.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1 - companion_fluxratio)
    F_EB = EB_fluxratio/(1 - EB_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    # calculate primary eclipse
    tm.set_data(time)
    k = R_EB/R_s
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    flux = tm.evaluate_ps(
        k=k,
        ldc=[float(u1), float(u2)],
        t0=0.0,
        p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=argp*(pi/180.)
        )
    # calculate secondary eclipse depth
    tm_sec.set_data(np.array([0.0]))
    sec_flux = tm_sec.evaluate_ps(
        k=1/k,
        ldc=[float(u1), float(u2)],
        t0=0.0, p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(argp+180)*(pi/180.)
        )
    # step 2: adjust the light curve to account for flux dilution
    # from EB and non-host star
    if companion_is_host:
        flux = (flux + F_EB/F_comp)/(1 + F_EB/F_comp)
        sec_flux = (sec_flux + F_comp/F_EB)/(1 + F_comp/F_EB)
        F_dilute = F_target/(F_comp + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    else:
        flux = (flux + F_EB/F_target)/(1 + F_EB/F_target)
        sec_flux = (sec_flux + F_target/F_EB)/(1 + F_target/F_EB)
        F_dilute = F_comp/(F_target + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    return flux, secdepth



def lnL_TP(time: np.ndarray, flux: np.ndarray, sigma: float, R_p: float,
           P_orb: float, inc: float, a: float, R_s: float,
           u1: float, u2: float, ecc: float, argp: float,
           companion_fluxratio: float = 0.0,
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
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
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
        ecc, argp,
        companion_fluxratio, companion_is_host
        )
    return 0.5*(np.sum((flux-model)**2 / sigma**2))


def lnL_EB(time: np.ndarray, flux: np.ndarray, sigma: float,
           R_EB: float, EB_fluxratio: float, P_orb: float, inc: float,
           a: float, R_s: float, u1: float, u2: float,
           ecc: float, argp: float,
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
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
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
        ecc, argp,
        companion_fluxratio, companion_is_host
        )
    if secdepth < 1.5*sigma:
        return 0.5*(np.sum((flux-model)**2 / sigma**2))
    else:
        return np.inf


def lnL_EB_twin(time: np.ndarray, flux: np.ndarray, sigma: float,
                R_EB: float, EB_fluxratio: float, P_orb: float,
                inc: float, a: float, R_s: float, u1: float, u2: float,
                ecc:float, argp: float,
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
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
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
        ecc, argp,
        companion_fluxratio, companion_is_host
        )
    return 0.5*(np.sum((flux-model)**2 / sigma**2))
