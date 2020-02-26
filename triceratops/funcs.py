import numpy as np
from pandas import DataFrame, read_csv
from astropy import constants
from scipy.interpolate import InterpolatedUnivariateSpline
from pkg_resources import resource_filename

Mass_nodes_Torres = np.array([0.26,0.47,0.59,0.69,0.87,0.98,1.085,1.4,1.65,2.0,2.5,3.0,4.4,15.0,40.0])
Teff_nodes_Torres = np.array([3170,3520,3840,4410,5150,5560,5940,6650,7300,8180,9790,11400,15200,30000,42000])
Rad_nodes_Torres = np.array([0.28,0.47,0.60,0.72,0.9,1.05,1.2,1.55,1.8,2.1,2.4,2.6,3.0,6.2,11.0])
Teff_all_Torres = np.linspace(4000, 42000, (42000-4000) + 1)
Rad_spline_Torres = InterpolatedUnivariateSpline(Teff_nodes_Torres, Rad_nodes_Torres)
Mass_spline_Torres = InterpolatedUnivariateSpline(Teff_nodes_Torres, Mass_nodes_Torres)
Rad_all_Torres = Rad_spline_Torres(Teff_all_Torres)
Mass_all_Torres = Mass_spline_Torres(Teff_all_Torres)
logg_all_Torres = np.log10((constants.G.cgs.value)*(Mass_all_Torres*constants.M_sun.cgs.value)/(Rad_all_Torres*constants.R_sun.cgs.value)**2)
L_all_Torres = 4*np.pi*constants.sigma_sb.cgs.value*(Rad_all_Torres*constants.R_sun.cgs.value)**2 * Teff_all_Torres**4 / constants.L_sun.cgs.value

Teff_nodes_cdwrf = np.array([2800, 3000, 3200, 3400, 3600, 3800, 4000])
Mass_nodes_cdwrf = np.array([0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63])
Rad_nodes_cdwrf = np.array([0.1, 0.17, 0.23, 0.35, 0.49, 0.58, 0.64])
logg_nodes_cdwrf = np.array([5.3, 5.15, 5.0, 4.85, 4.75, 4.68, 4.62])
L_nodes_cdwrf = np.array([0.001, 0.002, 0.006, 0.018, 0.035, 0.06, 0.11])
Teff_all_cdwrf = np.linspace(2800, 4000, (4000-2800) + 1)
Rad_spline_cdwrf = InterpolatedUnivariateSpline(Teff_nodes_cdwrf, Rad_nodes_cdwrf)
Mass_spline_cdwrf = InterpolatedUnivariateSpline(Teff_nodes_cdwrf, Mass_nodes_cdwrf)
logg_spline_cdwrf = InterpolatedUnivariateSpline(Teff_nodes_cdwrf, logg_nodes_cdwrf)
L_spline_cdwrf = InterpolatedUnivariateSpline(Teff_nodes_cdwrf, L_nodes_cdwrf)
Rad_all_cdwrf = Rad_spline_cdwrf(Teff_all_cdwrf)
Mass_all_cdwrf = Mass_spline_cdwrf(Teff_all_cdwrf)
logg_all_cdwrf = logg_spline_cdwrf(Teff_all_cdwrf)
L_all_cdwrf = L_spline_cdwrf(Teff_all_cdwrf)

def stellar_relations(Mass:float = 0.0, Rad:float = 0.0, Teff:float = 0.0, lum:float = 0.0):
	"""
	Estimates mass, radius, effective temperature, logg, and luminosity of a star given on of the quantities.
	Args:
		Mass (float): Star mass [Solar masses].
		Rad (float): Star radius [Solar radii].
		Teff (float): Star effective temperature [K].
		lum (float): Star luminosity [Solar luminosities].
	Returns:
		Mass (float): Star mass [Solar masses].
		Rad (float): Star radius [Solar radii].
		Teff (float): Star effective temperature [K].
		logg (float): Star logg.
		lum (float): Star luminosity [Solar luminosities].
	"""

	if (lum != 0.0 and lum > 0.095) or (Teff != 0.0 and Teff > 4000) or (Rad != 0.0 and Rad > 0.626) or (Mass != 0.0 and Mass > 0.63):
		Mass_all, Rad_all, Teff_all, logg_all, L_all = Mass_all_Torres, Rad_all_Torres, Teff_all_Torres, logg_all_Torres, L_all_Torres
	else:
		Mass_all, Rad_all, Teff_all, logg_all, L_all = Mass_all_cdwrf, Rad_all_cdwrf, Teff_all_cdwrf, logg_all_cdwrf, L_all_cdwrf

	if Mass != 0.0:
		idx = np.argmin(abs(Mass_all - Mass))
		return Mass_all[idx], Rad_all[idx], Teff_all[idx], logg_all[idx], L_all[idx]
	elif Rad != 0.0:
		idx = np.argmin(abs(Rad_all - Rad))
		return Mass_all[idx], Rad_all[idx], Teff_all[idx], logg_all[idx], L_all[idx]
	elif Teff != 0.0:
		idx = np.argmin(abs(Teff_all - Teff))
		return Mass_all[idx], Rad_all[idx], Teff_all[idx], logg_all[idx], L_all[idx]
	elif lum != 0.0:
		idx = np.argmin(abs(L_all - lum))
		return Mass_all[idx], Rad_all[idx], Teff_all[idx], logg_all[idx], L_all[idx]

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
		theta = 0.54042 + 0.23676*(V-Ks) - 0.00796*(V-Ks)**2
		Teff = 5040/theta
	elif V-Ks > 5.05:
		theta = -0.4809 + 0.8009*(V-Ks) - 0.1039*(V-Ks)**2 + 0.0056*(V-Ks)**3
		Teff = 5040/theta + 205.26
	return Teff

def nearby_star_luminosity(T1:float, T2:float, p1:float, p2:float, L1:float):
	"""
	Estimates luminosity of a nearby star using the TESS magnitude (T) and parallax (p)
	of the target star and the nearby star along with the the luminosity (L) of the target star.
	Args:
		T1 (float): TESS magnitude of target star.
		T2 (float): TESS magnitude of nearby star.
		p1 (float): Parallax of target star.
		p2 (float): Parallax of nearby star.
		L1 (float): Luminosity of target star.
	Returns:
		L2 (float): Luminosity of nearby star.
	"""
	L2 = L1 * 10**( (T1-T2)/2.5 + 2*np.log10(p1/p2) )
	return L2

# load ld coefficients 
LDC_FILE = resource_filename('triceratops','data/ldc.tsv')
ldc = read_csv(LDC_FILE, sep='\t', skiprows=48, usecols=[0,1,2,3,4,5])[2:]
ldc = DataFrame(ldc, dtype=float)
Zs = np.array(list(dict.fromkeys(ldc['Z'])))

def quad_coeffs(Teff:float, logg:float, Z:float):
	"""
	Returns quadratic limb darkening coefficients given stellar properties.
	Args:
		Teff (float): Star effective temperature [K].
		logg (float): Star logg.
		z (float): Star metallicity.
	Returns:
		u1 (float): 1st coefficient in quadratic limb darkening law.
		u2 (float): 2nd coefficient in quadratic limb darkening law.
	"""

	df_at_Z = ldc[ldc['Z'] == Zs[np.argmin(np.abs(Zs-Z))]]
	Teffs = np.array(list(dict.fromkeys(df_at_Z['Teff'])))
	loggs = np.array(list(dict.fromkeys(df_at_Z['logg'])))
	df = ldc[(ldc['Teff'] == Teffs[np.argmin(np.abs(Teffs-Teff))]) & (ldc['logg'] == loggs[np.argmin(np.abs(loggs-logg))])]
	# find corresponding coefficients 
	u1s, u2s = df["aLSM"], df["bLSM"]
	u1, u2 = np.mean(df["aLSM"].values), np.mean(df["bLSM"].values)
	return u1, u2

def renorm_flux(flux, companion_fluxratio:float, compaion:bool):
	"""
	Renormalizes light curve flux to account for the existence of an unresolved companion.
	Args:
		flux (numpy array): Normalized flux of each data point.
		companion_fluxratio (float): Proportion of flux that comes from the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
	Returns:
		renormed_flux (nump array): Remormalized flux of each data point.
	"""

	if compaion == True:
		renormed_flux = (flux - (1 - companion_fluxratio)) / companion_fluxratio
	elif compaion == False:
		renormed_flux = (flux - companion_fluxratio) / (1 - companion_fluxratio)
	return renormed_flux

def renorm_flux_err(flux_err, companion_fluxratio:float, compaion:bool):
	"""
	Renormalizes light curve flux error to account for the existence of an unresolved companion.
	Args:
		flux_err (numpy array): Normalized flux error of each data point.
		companion_fluxratio (float): Proportion of flux that comes from the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
	Returns:
		renormed_flux_err (nump array): Remormalized flux error of each data point.
	"""

	if compaion == True:
		renormed_flux_err = flux_err / companion_fluxratio
	elif compaion == False:
		renormed_flux_err = flux_err / (1 - companion_fluxratio)
	return renormed_flux_err

def adjust_flux_for_EB(flux, EB_fluxratio):
	"""
	Adjusts light curve to account for the existence of an EB.
	Args:
		flux (numpy array): Normalized flux of each data point.
		EB_fluxratio (float): Proportion of flux that comes from the EB.
	Returns:
		Adjusted flux (numpy array).
	"""

	return (flux - 1)*(1 - EB_fluxratio) + 1

def contrast_curve(path:str, Delta_mag:float):
	"""
	Determines separation at which a companion with a given Delta_mag can be ruled out. If no file provided, returns 2.2 arcseconds.
	Args:
		path (str): Path to contrast curve text file. File should contain column with separations (in arcsec)
					followed by column with Delta_mags.
		Delta_mag (float): Delta_mag at which we want to find a separation.
	Returns:
		sep (float): Separation (in arcseconds) that a companion can be ruled out at Delta_mag.
	"""
	if path is not None:
		data = np.loadtxt(path, delimiter=',')
		seps = data.T[0]
		Delta_mags = data.T[1]
		# don't let the smallest separation be 0 arcseconds
		if seps[0] == 0.0:
			seps[0] = seps[1]/10
		sep = np.interp(Delta_mag, Delta_mags, seps)
		if sep < 2.2:
			return sep
		else:
			return 2.2
	else:
		return 2.2