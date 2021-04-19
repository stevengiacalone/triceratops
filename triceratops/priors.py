import numpy as np
from scipy.stats import beta, powerlaw
from astropy import constants
from .funcs import (file_to_contrast_curve,
                   separation_at_contrast,
                   trilegal_results)

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi


def sample_rp(x, M_s):
    """
    Samples planet radii that are dependent on host mass.
    Args:
        x (numpy array): Random numbers between 0 and 1.
        M_s (numpy array): Host star masses [Solar radii].
    Returns:
        x (numpy array): Sampled planet radii [Earth radii].

    """
    R_break1 = 3.0
    R_break2 = 6.0
    R_min = 0.5
    R_max = 20.0
    # power coefficients for M > 0.45
    p1 = 0.0
    p2 = -4.0
    p3 = -0.5
    # power coefficients for M <= 0.45
    p4 = 0.0
    p5 = -7.0
    p6 = -0.5
    # normalizing constants  for M > 0.45
    A1 = R_break1**p1 / R_break1**p2
    A2 = R_break2**p2 / R_break2**p3
    I1 = (R_break1**(p1+1) - R_min**(p1+1))/(p1+1)
    I2 = A1*(R_break2**(p2+1) - R_break1**(p2+1))/(p2+1)
    I3 = A2*A1*(R_max**(p3+1) - R_break2**(p3+1))/(p3+1)
    Norm1 = 1/(I1+I2+I3)
    # normalizing constants  for M <= 0.45
    A3 = R_break1**p4 / R_break1**p5
    A4 = R_break2**p5 / R_break2**p6
    I4 = (R_break1**(p4+1) - R_min**(p4+1))/(p4+1)
    I5 = A3*(R_break2**(p5+1) - R_break1**(p5+1))/(p5+1)
    I6 = A4*A3*(R_max**(p6+1) - R_break2**(p6+1))/(p6+1)
    Norm2 = 1/(I4+I5+I6)

    mask1 = (
        (x <= Norm1*I1)
        & (M_s > 0.45)
        )
    mask2 = (
        (x > Norm1*I1)
        & (x <= Norm1*(I1+I2))
        & (M_s > 0.45)
        )
    mask3 = (
        (x > Norm1*(I1+I2))
        & (x <= Norm1*(I1+I2+I3))
        & (M_s > 0.45)
        )
    mask4 = (
        (x <= Norm2*I4)
        & (M_s <= 0.45)
        )
    mask5 = (
        (x > Norm2*I4)
        & (x <= Norm2*(I4+I5))
        & (M_s <= 0.45)
        )
    mask6 = (
        (x > Norm2*(I4+I5))
        & (x <= Norm2*(I4+I5+I6))
        & (M_s <= 0.45)
        )
    x[mask1] = (
        (x[mask1]/Norm1*(p1+1) + R_min**(p1+1))**(1/(p1+1))
        )
    x[mask2] = (
        (
            (x[mask2]/Norm1 - I1)*(p2+1)/A1
            + R_break1**(p2+1)
        )**(1/(p2+1))
        )
    x[mask3] = (
        (
            (x[mask3]/Norm1 - I1 - I2)*(p3+1)/(A1*A2)
            + R_break2**(p3+1)
        )**(1/(p3+1))
        )
    x[mask4] = (
        (x[mask4]/Norm2*(p4+1) + R_min**(p4+1))**(1/(p4+1))
        )
    x[mask5] = (
        (
            (x[mask5]/Norm2 - I4)*(p5+1)/A3
            + R_break1**(p5+1)
        )**(1/(p5+1))
        )
    x[mask6] = (
        (
            (x[mask6]/Norm2 - I4 - I5)*(p6+1)/(A3*A4)
            + R_break2**(p6+1)
        )**(1/(p6+1))
        )
    return x


def sample_inc(x, lower=0, upper=90):
    """
    Samples inclinations.
    Args:
        x (numpy array): Random numbers between 0 and 1.
        lower (float): Lower bound of inclinations [deg].
        Upper (float): Upper bound of inclinations [deg].
    Returns:
        x (numpy array): Sampled inclinations [deg].
    """
    # normalizing constant
    Norm = 1/(np.cos(lower*np.pi/180) - np.cos(upper*np.pi/180))
    x = np.arccos(np.cos(lower*np.pi/180) - x/Norm) * 180/np.pi
    return x

def sample_ecc(x, planet, P_orb):
    """
    Samples eccentricities using a Beta distribution
    for planets (see Kipping 2013) and a power law
    distirbution for binaries (see Moe & Di Stefano 2017).
    Args:
        x (numpy array): Random numbers between 0 and 1.
        planet (bool): True if planet and False if binary.
        P_orb (float): Orbital period.
    Returns:
        x (numpy array): Sampled eccentricities.
    """
    size = len(x)
    if planet == True:
        x = beta.rvs(0.867, 3.030, size=size)
    else:
        if P_orb <= 10:
            # note the variable is nu+1, not nu
            x = powerlaw.rvs(0.2, size=size)
        else:
            x = powerlaw.rvs(0.6, size=size)
    return x

def sample_w(x):
    """
    Samples argument of periastron.
    Args:
        x (numpy array): Random numbers between 0 and 1.
    Returns:
        x (numpy array): Sampled args of periastron [deg].
    """
    x = x*360
    return x

def sample_q(x, M_s):
    """
    Samples mass ratios of short-period binaries.
    Args:
        x (numpy array): Random numbers between 0 and 1.
        M_s (float): Primary star mass [Solar masses].
    Returns:
        x (numpy array): Sampled mass ratios.
    """
    if M_s >= 1.0:
        # power coefficients
        p1 = 0.3
        p2 = -0.5
        # normalizing constants
        # continuity between first two segments
        A1 = (0.3**p1) / (0.3**p2)
        # satisfy F_twin condition
        F_twin = 0.30
        A2 = (
            1 + (F_twin)/(1-F_twin)
            * ((1.0**(p2+1) - 0.3**(p2+1))/(p2+1))
            / ((1.0**(p2+1) - 0.95**(p2+1))/(p2+1))
            )
        I1 = (0.3**(p1+1) - 0.1**(p1+1))/(p1+1)
        I2 = A1*(0.95**(p2+1) - 0.3**(p2+1))/(p2+1)
        I3 = A2*A1*(1.0**(p2+1) - 0.95**(p2+1))/(p2+1)
        Norm = 1/(I1+I2+I3)

        mask1 = x <= Norm*I1
        mask2 = (x > Norm*I1) & (x <= Norm*(I1+I2))
        mask3 = (x > Norm*(I1+I2)) & (x <= Norm*(I1+I2+I3))
        x[mask1] = (
            (x[mask1]/Norm*(p1+1) + 0.1**(p1+1))**(1/(p1+1))
            )
        x[mask2] = (
            ((x[mask2]/Norm - I1)*(p2+1)/A1 + 0.3**(p2+1))**(1/(p2+1))
            )
        x[mask3] = (
            (
                (x[mask3]/Norm - I1 - I2)*(p2+1)/(A1*A2)
                + 0.95**(p2+1))**(1/(p2+1))
            )
    elif (M_s < 1.0) & (M_s >= 0.3):
        # q lower limit (to prevent M_EB < 0.1 solar masses)
        q_min = 0.1/M_s
        # power coefficients
        p1 = 0.3
        p2 = -0.5
        # normalizing constants
        # continuity between first two segments
        A1 = (0.3**p1) / (0.3**p2)
        # satisfy F_twin condition
        F_twin = 0.30
        A2 = (
            1 + (F_twin)/(1-F_twin)
            * ((1.0**(p2+1) - 0.3**(p2+1))/(p2+1))
            / ((1.0**(p2+1) - 0.95**(p2+1))/(p2+1))
            )
        I1 = (0.3**(p1+1) - q_min**(p1+1))/(p1+1)
        I2 = A1*(0.95**(p2+1) - 0.3**(p2+1))/(p2+1)
        I3 = A2*A1*(1.0**(p2+1) - 0.95**(p2+1))/(p2+1)
        Norm = 1/(I1+I2+I3)

        mask1 = x <= Norm*I1
        mask2 = (x > Norm*I1) & (x <= Norm*(I1+I2))
        mask3 = (x > Norm*(I1+I2)) & (x <= Norm*(I1+I2+I3))
        x[mask1] = (
            (x[mask1]/Norm*(p1+1) + q_min**(p1+1))**(1/(p1+1))
            )
        x[mask2] = (
            ((x[mask2]/Norm - I1)*(p2+1)/A1 + 0.3**(p2+1))**(1/(p2+1))
            )
        x[mask3] = (
            (
                (x[mask3]/Norm - I1 - I2)*(p2+1)/(A1*A2)
                + 0.95**(p2+1))**(1/(p2+1))
            )
    elif (M_s < 0.3) & (M_s > 0.1):
        # q lower limit (to prevent M_EB < 0.1 solar masses)
        q_min = 0.1/M_s
        # power coefficients
        p2 = -0.5
        # normalizing constants
        # satisfy F_twin condition
        F_twin = 0.30
        A2 = (
            1 + (F_twin)/(1-F_twin)
            * ((1.0**(p2+1) - q_min**(p2+1))/(p2+1))
            / ((1.0**(p2+1) - 0.95**(p2+1))/(p2+1))
            )
        I2 = (0.95**(p2+1) - q_min**(p2+1))/(p2+1)
        I3 = A2*(1.0**(p2+1) - 0.95**(p2+1))/(p2+1)
        Norm = 1/(I2+I3)

        mask2 = x <= Norm*(I2)
        mask3 = (x > Norm*(I2)) & (x <= Norm*(I2+I3))
        x[mask2] = (
            (x[mask2]/Norm*(p2+1) + q_min**(p2+1))**(1/(p2+1))
            )
        x[mask3] = (
            (
                (x[mask3]/Norm - I2)*(p2+1)/(A2)
                + 0.95**(p2+1))**(1/(p2+1))
            )
    elif M_s <= 0.1:
        x = np.full(len(x), 1.0)
    return x


def sample_q_companion(x, M_s):
    """
    Samples mass ratios of long-period companions.
    Args:
        x (numpy array): Random numbers between 0 and 1.
        M_s (float): Primary star mass [Solar masses].
    Returns:
        x (numpy array): Sampled mass ratios.
    """
    if M_s >= 1.0:
        # power coefficients
        p1 = 0.3
        p2 = -0.95
        # normalizing constants
        # continuity between first two segments
        A1 = (0.3**p1) / (0.3**p2)
        # satisfy F_twin condition
        F_twin = 0.05
        A2 = (
            1 + (F_twin)/(1-F_twin)
            * ((1.0**(p2+1) - 0.3**(p2+1))/(p2+1))
            / ((1.0**(p2+1) - 0.95**(p2+1))/(p2+1))
            )
        I1 = (0.3**(p1+1) - 0.1**(p1+1))/(p1+1)
        I2 = A1*(0.95**(p2+1) - 0.3**(p2+1))/(p2+1)
        I3 = A2*A1*(1.0**(p2+1) - 0.95**(p2+1))/(p2+1)
        Norm = 1/(I1+I2+I3)

        mask1 = x <= Norm*I1
        mask2 = (x > Norm*I1) & (x <= Norm*(I1+I2))
        mask3 = (x > Norm*(I1+I2)) & (x <= Norm*(I1+I2+I3))
        x[mask1] = (
            (x[mask1]/Norm*(p1+1) + 0.1**(p1+1))**(1/(p1+1))
            )
        x[mask2] = (
            ((x[mask2]/Norm - I1)*(p2+1)/A1 + 0.3**(p2+1))**(1/(p2+1))
            )
        x[mask3] = (
            (
                (x[mask3]/Norm - I1 - I2)*(p2+1)/(A1*A2)
                + 0.95**(p2+1))**(1/(p2+1))
            )
    elif (M_s < 1.0) & (M_s >= 0.3):
        # q lower limit (to prevent M_comp < 0.1 solar masses)
        q_min = 0.1/M_s
        # power coefficients
        p1 = 0.3
        p2 = -0.95
        # normalizing constants
        # continuity between first two segments
        A1 = (0.3**p1) / (0.3**p2)
        # satisfy F_twin condition
        F_twin = 0.05
        A2 = (
            1 + (F_twin)/(1-F_twin)
            * ((1.0**(p2+1) - 0.3**(p2+1))/(p2+1))
            / ((1.0**(p2+1) - 0.95**(p2+1))/(p2+1))
            )
        I1 = (0.3**(p1+1) - q_min**(p1+1))/(p1+1)
        I2 = A1*(0.95**(p2+1) - 0.3**(p2+1))/(p2+1)
        I3 = A2*A1*(1.0**(p2+1) - 0.95**(p2+1))/(p2+1)
        Norm = 1/(I1+I2+I3)

        mask1 = x <= Norm*I1
        mask2 = (x > Norm*I1) & (x <= Norm*(I1+I2))
        mask3 = (x > Norm*(I1+I2)) & (x <= Norm*(I1+I2+I3))
        x[mask1] = (
            (x[mask1]/Norm*(p1+1) + q_min**(p1+1))**(1/(p1+1))
            )
        x[mask2] = (
            ((x[mask2]/Norm - I1)*(p2+1)/A1 + 0.3**(p2+1))**(1/(p2+1))
            )
        x[mask3] = (
            (
                (x[mask3]/Norm - I1 - I2)*(p2+1)/(A1*A2)
                + 0.95**(p2+1))**(1/(p2+1))
            )
    elif (M_s < 0.3) & (M_s > 0.1):
        # q lower limit (to prevent M_comp < 0.1 solar masses)
        q_min = 0.1/M_s
        # power coefficients
        p2 = -0.95
        # normalizing constants
        # satisfy F_twin condition
        F_twin = 0.05
        A2 = (
            1 + (F_twin)/(1-F_twin)
            * ((1.0**(p2+1) - q_min**(p2+1))/(p2+1))
            / ((1.0**(p2+1) - 0.95**(p2+1))/(p2+1))
            )
        I2 = (0.95**(p2+1) - q_min**(p2+1))/(p2+1)
        I3 = A2*(1.0**(p2+1) - 0.95**(p2+1))/(p2+1)
        Norm = 1/(I2+I3)

        mask2 = x <= Norm*(I2)
        mask3 = (x > Norm*(I2)) & (x <= Norm*(I2+I3))
        x[mask2] = (
            (x[mask2]/Norm*(p2+1) + q_min**(p2+1))**(1/(p2+1))
            )
        x[mask3] = (
            (
                (x[mask3]/Norm - I2)*(p2+1)/(A2)
                + 0.95**(p2+1))**(1/(p2+1))
            )
    elif M_s <= 0.1:
        x = np.full(len(x), 1.0)
    return x


def lnprior_Mstar_planet(M_s: np.array):
    """
    Estimates planet occurrence rate for all planet radii and orbital
    periods under 50 days and turns it into a prior probability.
    Args:
        M_s (float): Star mass [Solar masses].
    Returns:
        lnprior_planet (float): The log probability of there being a
                                short period planet
                                around a star of mass M_s.
    """
    f_p = np.zeros(len(M_s))
    mask = ((2.5 - 1.5*M_s) > 0.1)
    f_p[mask] = 2.5 - 1.5*M_s[mask]
    mask = ((2.5 - 1.5*M_s) <= 0.1)
    f_p[mask] = 0.1
    f_p[f_p > 1.0] = 1.0
    lnprior_Mstar = np.log(f_p)
    # return lnprior_Mstar as 0 (omitted due to bias)
    return 0.0


def lnprior_Mstar_binary(M_s: np.array):
    """
    Calculates the companion rate of the host star for periods
    under 50 days and turns it into a prior probability.
    Args:
        M_s (numpy array): Host star masses [solar masses].
    Returns:
        lnprior_EB (float): The log probability of there being
                            a short period binary
                            around a star of mass M_s.
    """
    f_comp = np.zeros(len(M_s))
    f1 = np.zeros(len(M_s))
    f2 = np.zeros(len(M_s))
    f3 = np.zeros(len(M_s))
    t1 = np.zeros(len(M_s))
    t2_partial = np.zeros(len(M_s))
    alpha = 0.018
    dlogP = 0.7
    max_Porb = 50

    mask = (M_s >= 1.0)
    f1[mask] = (
        0.020 + 0.04*np.log10(M_s[mask])
        + 0.07*(np.log10(M_s[mask]))**2
        )
    f2[mask] = (
        0.039 + 0.07*np.log10(M_s[mask])
        + 0.01*(np.log10(M_s[mask]))**2
        )
    f3[mask] = (
        0.078 - 0.05*np.log10(M_s[mask])
        + 0.04*(np.log10(M_s[mask]))**2
        )
    t1[mask] = f1[mask]
    t2_partial[mask] = (
        0.5*(np.log10(max_Porb) - 1.0)
        * (
            2.0*f1[mask]+(f2[mask]-f1[mask]-alpha*dlogP)
            * (np.log10(max_Porb) - 1.0)
            )
        )
    f_comp[mask] = t1[mask] + t2_partial[mask]

    mask = (M_s < 1.0)
    f1[mask] = (
        0.020 + 0.04*np.log10(1.0)
        + 0.07*(np.log10(1.0))**2
        )
    f2[mask] = (
        0.039 + 0.07*np.log10(1.0)
        + 0.01*(np.log10(1.0))**2
        )
    f3[mask] = (
        0.078 - 0.05*np.log10(1.0)
        + 0.04*(np.log10(1.0))**2
        )
    t1[mask] = f1[mask]
    t2_partial[mask] = (
        0.5*(np.log10(max_Porb) - 1.0)
        * (
            2.0*f1[mask]+(f2[mask]-f1[mask]-alpha*dlogP)
            * (np.log10(max_Porb) - 1.0)
            )
        )
    f_comp[mask] = t1[mask] + t2_partial[mask]
    f_comp[mask] = 0.65*f_comp[mask]+0.35*f_comp[mask]*M_s[mask]

    f_comp[f_comp > 1.0] = 1.0
    lnprior_Mstar = np.log(f_comp)
    # return lnprior_Mstar (omitted due to bias)
    return 0.0


def lnprior_Porb_planet(P_orb: float):
    """
    Calculates probability of a planet with a given
    orbital period < 50 days.
    Args:
        P_orb (float): Orbital period [days].
    Returns:
        lnprior_Porb (float): Log probability of planet having an
                              orbital period P_orb +/- 0.1 days.
    """
    P_break = 10
    P_min = 0.1
    P_max = 50
    p1 = 1.5
    p2 = 0.0
    A = P_break**p1 / P_break**p2
    I1 = (P_break**(p1+1) - P_min**(p1+1))/(p1+1)
    I2 = A*(P_max**(p2+1) - P_break**(p2+1))/(p2+1)
    Norm = 1/(I1+I2)

    if P_orb < P_min+0.1:
        P_orb = P_min+0.1
    elif P_orb > P_max-0.1:
        P_orb = P_max-0.1

    if P_orb <= P_break-0.1:
        I1 = ((P_orb+0.1)**(p1+1) - (P_orb-0.1)**(p1+1))/(p1+1)
        prob = Norm * I1
    elif P_orb >= P_break+0.1:
        I1 = A*((P_orb+0.1)**(p2+1) - (P_orb-0.1)**(p2+1))/(p2+1)
        prob = Norm * I1
    else:
        I1 = (P_break**(p1+1) - (P_orb-0.1)**(p1+1))/(p1+1)
        I2 = A*((P_orb+0.1)**(p2+1) - P_break**(p2+1))/(p2+1)
        prob = Norm * (I1+I2)

    lnprior_Porb = np.log(prob)
    return lnprior_Porb


def lnprior_Porb_binary(P_orb: float):
    """
    Calculates probability of a binary with a given
    orbital period < 50 days.
    Args:
        P_orb (float): Orbital period [days].
    Returns:
        lnprior_Porb (float): Log probability of binary having an
                              orbital period P_orb +/- 0.1 days.
    """
    P_break = 0.3
    P_min = 0.1
    P_max = 50
    p1 = 5.0
    p2 = 0.5

    A = P_break**p1 / P_break**p2
    I1 = (P_break**(p1+1) - P_min**(p1+1))/(p1+1)
    I2 = A*(P_max**(p2+1) - P_break**(p2+1))/(p2+1)
    Norm = 1/(I1+I2)

    if P_orb < P_min+0.1:
        P_orb = P_min+0.1
    elif P_orb > P_max-0.1:
        P_orb = P_max-0.1

    if P_orb <= P_break-0.1:
        I1 = ((P_orb+0.1)**(p1+1) - (P_orb-0.1)**(p1+1))/(p1+1)
        prob = Norm * I1
    elif P_orb >= P_break+0.1:
        I1 = A*((P_orb+0.1)**(p2+1) - (P_orb-0.1)**(p2+1))/(p2+1)
        prob = Norm * I1
    else:
        I1 = (P_break**(p1+1) - (P_orb-0.1)**(p1+1))/(p1+1)
        I2 = A*((P_orb+0.1)**(p2+1) - P_break**(p2+1))/(p2+1)
        prob = Norm * (I1+I2)

    lnprior_Porb = np.log(prob)
    return lnprior_Porb


def lnprior_bound_TP(M_s: float, plx: float, delta_mags: np.array,
                     separations: np.array, contrasts: np.array):
    """
    Calculates the bound companion rate of the target star,
    assuming the orbital period of the companion is > 2500 days.
    (Moe & Kratter 2020 show large planet suppression for S-type
    planets with short periods when the two stars have a < 10 au.)
    Equations from Moe & DiStefano 2017.
    Args:
        M_s (float): Target star mass [solar masses].
        plx (float): Parallax of the target star [mas].
        delta_mags (numpy array): Contrasts of simulated
                                  companions (delta_mag).
        separations (numpy array): Separation at contrast (arcsec).
        contrasts (numpy array): Contrast at separation (delta_mag).
    Returns:
        lnprior_bound (float): The log probability of there being
                               a bound companion.
    """
    # determine maximum physical separations based on angular
    # separation constraints and parallax
    if np.isnan(plx):
        plx = 0.1
    d = 1000/plx
    seps = d*separation_at_contrast(delta_mags, separations, contrasts)

    # calculate prior probability for M_s >= 1.0 solar masses
    if M_s >= 1.0:
        f1 = 0.020 + 0.04*np.log10(M_s) + 0.07*(np.log10(M_s))**2
        f2 = 0.039 + 0.07*np.log10(M_s) + 0.01*(np.log10(M_s))**2
        f3 = 0.078 - 0.05*np.log10(M_s) + 0.04*(np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4*pi**2)/(G*M_s*Msun)*(seps*au)**3)**(1/2)/86400

        t1 = f1
        t2_partial = (
            0.5*(np.log10(max_Porbs) - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (np.log10(max_Porbs) - 1.0)
                )
            )
        t2 = (
            0.5*(2.0 - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (2.0 - 1.0)
                )
            )
        t3_partial = (
            0.5*alpha
            * (
                np.log10(max_Porbs)**2-5.4
                * np.log10(max_Porbs)+6.8
                )
            + f2*(np.log10(max_Porbs) - 2.0)
            )
        t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
        t4_partial = (
            alpha*dlogP*(np.log10(max_Porbs) - 3.4)
            + f2*(np.log10(max_Porbs) - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (
                0.238095*np.log10(max_Porbs)**2
                - 0.952381*np.log10(max_Porbs)
                + 0.485714
                )
            )
        t4 = (
            alpha*dlogP*(5.5 - 3.4) + f2*(5.5 - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
            )
        t5_partial = (
            f3*(3.33333 - 17.3566*np.exp(-0.3*np.log10(max_Porbs)))
            )
        t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*8.0))

        f_comp = np.zeros(len(seps))
        mask = (np.log10(max_Porbs) < 1.0)
        # f_comp[mask] = t1
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 1.0)
            & (np.log10(max_Porbs) < 2.0)
            )
        # f_comp[mask] = t1 + t2_partial[mask]
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 2.0)
            & (np.log10(max_Porbs) < 3.4)
            )
        # f_comp[mask] = t1 + t2 + t3_partial[mask]
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 3.4)
            & (np.log10(max_Porbs) < 5.5)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4_partial[mask]
        f_comp[mask] = t4_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 5.5)
            & (np.log10(max_Porbs) < 8.0)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5_partial[mask]
        f_comp[mask] = t4 + t5_partial[mask]
        mask = (np.log10(max_Porbs) >= 8.0)
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5
        f_comp[mask] = t4 + t5
        lnprior_bound = np.log(f_comp)

    # calculate prior probability for M_s < 1.0 solar masses
    else:
        M_act = M_s
        M_s = 1.0
        f1 = 0.020 + 0.04*np.log10(M_s) + 0.07*(np.log10(M_s))**2
        f2 = 0.039 + 0.07*np.log10(M_s) + 0.01*(np.log10(M_s))**2
        f3 = 0.078 - 0.05*np.log10(M_s) + 0.04*(np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4*pi**2)/(G*M_s*Msun)*(seps*au)**3)**(1/2)/86400

        t1 = f1
        t2_partial = (
            0.5*(np.log10(max_Porbs) - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (np.log10(max_Porbs) - 1.0)
                )
            )
        t2 = (
            0.5*(2.0 - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (2.0 - 1.0)
                )
            )
        t3_partial = (
            0.5*alpha
            * (
                np.log10(max_Porbs)**2
                - 5.4*np.log10(max_Porbs) + 6.8
                )
            + f2*(np.log10(max_Porbs) - 2.0)
            )
        t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
        t4_partial = (
            alpha*dlogP*(np.log10(max_Porbs) - 3.4)
            + f2*(np.log10(max_Porbs) - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (
                0.238095*np.log10(max_Porbs)**2
                - 0.952381*np.log10(max_Porbs) + 0.485714
                )
            )
        t4 = (
            alpha*dlogP*(5.5 - 3.4)
            + f2*(5.5 - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
            )
        t5_partial = (
            f3*(3.33333 - 17.3566*np.exp(-0.3*np.log10(max_Porbs)))
            )
        t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*8.0))

        f_comp = np.zeros(len(seps))
        mask = (np.log10(max_Porbs) < 1.0)
        # f_comp[mask] = t1
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 1.0)
            & (np.log10(max_Porbs) < 2.0)
            )
        # f_comp[mask] = t1 + t2_partial[mask]
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 2.0)
            & (np.log10(max_Porbs) < 3.4)
            )
        # f_comp[mask] = t1 + t2 + t3_partial[mask]
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 3.4)
            & (np.log10(max_Porbs) < 5.5)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4_partial[mask]
        f_comp[mask] = t4_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 5.5)
            & (np.log10(max_Porbs) < 8.0)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5_partial[mask]
        f_comp[mask] = t4 + t5_partial[mask]
        mask = (np.log10(max_Porbs) >= 8.0)
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5
        f_comp[mask] = t4 + t5
        f_act = 0.65*f_comp+0.35*f_comp*M_act
        lnprior_bound = np.log(f_act)

    return lnprior_bound

def lnprior_bound_EB(M_s: float, plx: float, delta_mags: np.array,
                     separations: np.array, contrasts: np.array):
    """
    Calculates the bound companion rate of the target star for EB
    scenarios. Assumes the period of the tertiary is > 10 days.
    Equations from Moe and DiStefano 2017.
    Args:
        M_s (float): Target star mass [solar masses].
        plx (float): Parallax of the target star [mas].
        delta_mags (numpy array): Contrasts of simulated
                                  companions (delta_mag).
        separations (numpy array): Separation at contrast (arcsec).
        contrasts (numpy array): Contrast at separation (delta_mag).
    Returns:
        lnprior_bound (float): The log probability of there being
                               a bound companion.
    """
    # determine maximum physical separations based on angular
    # separation constraints and parallax
    if np.isnan(plx):
        plx = 0.1
    d = 1000/plx
    seps = d*separation_at_contrast(delta_mags, separations, contrasts)

    # calculate prior probability for M_s >= 1.0 solar masses
    if M_s >= 1.0:
        f1 = 0.020 + 0.04*np.log10(M_s) + 0.07*(np.log10(M_s))**2
        f2 = 0.039 + 0.07*np.log10(M_s) + 0.01*(np.log10(M_s))**2
        f3 = 0.078 - 0.05*np.log10(M_s) + 0.04*(np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4*pi**2)/(G*M_s*Msun)*(seps*au)**3)**(1/2)/86400

        # t1 = f1
        t2_partial = (
            0.5*(np.log10(max_Porbs) - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (np.log10(max_Porbs) - 1.0)
                )
            )
        t2 = (
            0.5*(2.0 - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (2.0 - 1.0)
                )
            )
        t3_partial = (
            0.5*alpha
            * (
                np.log10(max_Porbs)**2-5.4
                * np.log10(max_Porbs)+6.8
                )
            + f2*(np.log10(max_Porbs) - 2.0)
            )
        t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
        t4_partial = (
            alpha*dlogP*(np.log10(max_Porbs) - 3.4)
            + f2*(np.log10(max_Porbs) - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (
                0.238095*np.log10(max_Porbs)**2
                - 0.952381*np.log10(max_Porbs)
                + 0.485714
                )
            )
        t4 = (
            alpha*dlogP*(5.5 - 3.4) + f2*(5.5 - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
            )
        t5_partial = (
            f3*(3.33333 - 17.3566*np.exp(-0.3*np.log10(max_Porbs)))
            )
        t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*8.0))

        f_comp = np.zeros(len(seps))
        mask = (np.log10(max_Porbs) < 1.0)
        # f_comp[mask] = t1
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 1.0)
            & (np.log10(max_Porbs) < 2.0)
            )
        # f_comp[mask] = t1 + t2_partial[mask]
        f_comp[mask] = t2_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 2.0)
            & (np.log10(max_Porbs) < 3.4)
            )
        # f_comp[mask] = t1 + t2 + t3_partial[mask]
        f_comp[mask] = t2 + t3_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 3.4)
            & (np.log10(max_Porbs) < 5.5)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4_partial[mask]
        f_comp[mask] = t2 + t3 + t4_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 5.5)
            & (np.log10(max_Porbs) < 8.0)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5_partial[mask]
        f_comp[mask] = t2 + t3 + t4 + t5_partial[mask]
        mask = (np.log10(max_Porbs) >= 8.0)
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5
        f_comp[mask] = t2 + t3 + t4 + t5
        lnprior_bound = np.log(f_comp)

    # calculate prior probability for M_s < 1.0 solar masses
    else:
        M_act = M_s
        M_s = 1.0
        f1 = 0.020 + 0.04*np.log10(M_s) + 0.07*(np.log10(M_s))**2
        f2 = 0.039 + 0.07*np.log10(M_s) + 0.01*(np.log10(M_s))**2
        f3 = 0.078 - 0.05*np.log10(M_s) + 0.04*(np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4*pi**2)/(G*M_s*Msun)*(seps*au)**3)**(1/2)/86400

        t1 = f1
        t2_partial = (
            0.5*(np.log10(max_Porbs) - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (np.log10(max_Porbs) - 1.0)
                )
            )
        t2 = (
            0.5*(2.0 - 1.0)
            * (
                2.0*f1 + (f2 - f1 - alpha*dlogP)
                * (2.0 - 1.0)
                )
            )
        t3_partial = (
            0.5*alpha
            * (
                np.log10(max_Porbs)**2
                - 5.4*np.log10(max_Porbs) + 6.8
                )
            + f2*(np.log10(max_Porbs) - 2.0)
            )
        t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
        t4_partial = (
            alpha*dlogP*(np.log10(max_Porbs) - 3.4)
            + f2*(np.log10(max_Porbs) - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (
                0.238095*np.log10(max_Porbs)**2
                - 0.952381*np.log10(max_Porbs) + 0.485714
                )
            )
        t4 = (
            alpha*dlogP*(5.5 - 3.4)
            + f2*(5.5 - 3.4)
            + (f3 - f2 - alpha*dlogP)
            * (0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
            )
        t5_partial = (
            f3*(3.33333 - 17.3566*np.exp(-0.3*np.log10(max_Porbs)))
            )
        t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*8.0))

        f_comp = np.zeros(len(seps))
        mask = (np.log10(max_Porbs) < 1.0)
        # f_comp[mask] = t1
        f_comp[mask] = 0.0
        mask = (
            (np.log10(max_Porbs) >= 1.0)
            & (np.log10(max_Porbs) < 2.0)
            )
        # f_comp[mask] = t1 + t2_partial[mask]
        f_comp[mask] = t2_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 2.0)
            & (np.log10(max_Porbs) < 3.4)
            )
        # f_comp[mask] = t1 + t2 + t3_partial[mask]
        f_comp[mask] = t2 + t3_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 3.4)
            & (np.log10(max_Porbs) < 5.5)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4_partial[mask]
        f_comp[mask] = t2 + t3 + t4_partial[mask]
        mask = (
            (np.log10(max_Porbs) >= 5.5)
            & (np.log10(max_Porbs) < 8.0)
            )
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5_partial[mask]
        f_comp[mask] = t2 + t3 + t4 + t5_partial[mask]
        mask = (np.log10(max_Porbs) >= 8.0)
        # f_comp[mask] = t1 + t2 + t3 + t4 + t5
        f_comp[mask] = t2 + t3 + t4 + t5
        f_act = 0.65*f_comp+0.35*f_comp*M_act
        lnprior_bound = np.log(f_act)

    return lnprior_bound

def lnprior_background(N_comp: int, delta_mags: np.array,
                       separations: np.array,
                       contrasts: np.array):
    """
    Calculates the limiting separation (in arcsecs)
    at a given delta_mag.
    Args:
        N_comp (int): Number of stars obtained from
                      trilegal simulation.
        delta_mags (numpy array): Contrasts of simulated
                                  companions (delta_mag).
        separations (numpy array): Separation at contrast (arcsec).
        contrasts (numpy array): Contrast at separation (delta_mag).
    Returns:
        lnprior_bg (float): The log probability of there being
                            a background star.
    """
    seps = separation_at_contrast(delta_mags, separations, contrasts)
    lnprior_bg = np.log10((N_comp/0.1) * (1/3600)**2 * seps**2)
    return lnprior_bg
