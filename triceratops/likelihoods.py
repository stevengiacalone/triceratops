import numpy as np
# ignore invalid floating-point operations, doesn't affect calculation
np.seterr(invalid="ignore")
from astropy import constants
from scipy.optimize import minimize
import batman

from .funcs import stellar_relations, quad_coeffs, adjust_flux_for_EB, renorm_flux, renorm_flux_err 

def simulate_TP_transit(x0:np.ndarray, time:np.ndarray, P_orb:float, a:float, R_s:float, u1:float, u2:float):
	"""
	Simulates a transiting planet light curve using BATMAN.
	Args:
		x0 (numpy array): Inclindation [degrees] and planet radius [Earth radii].
		time (numpy array): Time of each data point [days from transit midpoint].
		P_orb (float): Orbital period [days].
		a (float): Semimajor axis [cm].
		R_s (float): Star radius [Solar radii].
		u1 (float): 1st coefficient in quadratic limb darkening law.
		u2 (float): 2nd coefficient in quadratic limb darkening law.
	Returns:
		m.light_curve (numpy array): Normalized flux at eat time given.
	"""

	inc = x0[0]
	R_p = x0[1]
	
	if (R_p*constants.R_earth.cgs.value + R_s*constants.R_sun.cgs.value) >= a:
		return np.ones(len(time))

	params = batman.TransitParams()
	params.t0 = 0.0          
	params.per = P_orb                  
	params.rp = R_p*constants.R_earth.cgs.value/(R_s*constants.R_sun.cgs.value)
	params.a = a/(R_s*constants.R_sun.cgs.value)          
	params.inc = inc             
	params.ecc = 0.0                    
	params.w = 0.0                   
	params.u = [u1, u2]                
	params.limb_dark = "quadratic"     

	m = batman.TransitModel(params, time)    
	return m.light_curve(params)

def simulate_EB_transit(x0:np.ndarray, time:np.ndarray, P_orb:float, lum:float):
	"""
	Simulates a eclipsing binary light curve using BATMAN.
	Args:
		x0 (numpy array): Inclindation [degrees] and proportion of flux contributed by the EB.
		time (numpy array): Time of each data point [days from transit midpoint].
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
	Returns:
		m.light_curve (numpy array): Normalized flux at eat time given.
	"""

	inc = x0[0]
	EB_fluxratio = x0[1]
	
	# primary properties
	M_s, R_s, Teff, logg, primary_lum = stellar_relations(lum=lum*(1-EB_fluxratio))
	u1, u2 = quad_coeffs(Teff, logg, 0.0)
	# EB properties
	M_EB, R_EB, Teff_EB, logg_EB, lum_EB = stellar_relations(lum=lum*EB_fluxratio)
	a = ((constants.G.cgs.value*(M_s+M_EB)*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)

	if (R_EB *constants.R_sun.cgs.value + R_s*constants.R_sun.cgs.value) >= a:
		return np.ones(len(time))

	params = batman.TransitParams()
	params.t0 = 0.0          
	params.per = P_orb                  
	params.rp = R_EB/R_s
	params.a = a/(R_s*constants.R_sun.cgs.value)          
	params.inc = inc             
	params.ecc = 0.0                    
	params.w = 0.0                   
	params.u = [u1, u2]                
	params.limb_dark = "quadratic"     

	m = batman.TransitModel(params, time) 
	adjusted_flux = adjust_flux_for_EB(m.light_curve(params), EB_fluxratio)
	return adjusted_flux

def log_likelihood_TP(x0:np.ndarray, time:np.ndarray, flux:np.ndarray, flux_err:np.ndarray,  P_orb:float, a:float, R_s:float, u1:float, u2:float, companion_fluxratio:float = 0.0, companion:bool = False):
	"""
	Calculates the log likelihood of a transiting planet scenario by comparing a simulated light curve and the TESS light curve.
	Args:
		x0 (numpy array): Inclindation [degrees] and planet radius [Earth radii].
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		a (float): Semimajor axis [cm].
		R_s (float): Star radius [Solar radii].
		u1 (float): 1st coefficient in quadratic limb darkening law.
		u2 (float): 2nd coefficient in quadratic limb darkening law.
		companion_fluxratio (float): Proportion of flux provided by the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
	Returns:
		Log likelihood (float).
	"""

	model = simulate_TP_transit(x0, time, P_orb, a, R_s, u1, u2)
	renormed_flux = renorm_flux(flux, companion_fluxratio, companion) 
	renormed_flux_err = renorm_flux_err(flux_err, companion_fluxratio, companion)

	if (np.unique(model).size == 1) or (x0[0] > 90) or np.logical_not(0 < x0[1] < 33):
		return np.inf
	else:
		return 0.5*(np.sum((renormed_flux-model)**2 / renormed_flux_err**2))

def log_likelihood_EB(x0:np.ndarray, time:np.ndarray, flux:np.ndarray, flux_err:np.ndarray,  P_orb:float, lum:float, companion_fluxratio:float = 0.0, companion:bool = False):
	"""
	Calculates the log likelihood of an eclipsing binary scenario by comparing a simulated light curve and the TESS light curve.
	Args:
		x0 (numpy array): Inclindation [degrees] and planet radius [Earth radii].
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		companion_fluxratio (float): Proportion of flux provided by the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
	Returns:
		Log likelihood (float).
	"""

	model = simulate_EB_transit(x0, time, P_orb, lum)
	renormed_flux = renorm_flux(flux, companion_fluxratio, companion) 
	renormed_flux_err = renorm_flux_err(flux_err, companion_fluxratio, companion)

	if (np.unique(model).size == 1) or (x0[0] > 90) or np.logical_not(0.001/lum <= x0[1] <= 0.5):
		return np.inf
	else:
		return 0.5*(np.sum((renormed_flux-model)**2 / renormed_flux_err**2))

def grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, N, M, inc_min, inc_max, R_p_min, R_p_max, companion_fluxratio:float = 0.0, companion:bool = False, delta_inc = None, inc_guess = None, delta_R_p = None, R_p_guess = None):
	"""
	Makes an initial guess at the most likely transiting planet scenario using a grid of inclinations and planet radii.
	Args:
		x0 (numpy array): Inclindation [degrees] and planet radius [Earth radii].
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		a (float): Semimajor axis [cm].
		R_s (float): Star radius [Solar radii].
		u1 (float): 1st coefficient in quadratic limb darkening law.
		u2 (float): 2nd coefficient in quadratic limb darkening law.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 
		inc_min (float): Minimum inclination considered.
		inc_max (float): Maximum inclination considered.
		R_p_min (float): Minimum planet radius considered.
		R_p_max (float): Maximum planet radius considered.
		companion_fluxratio (float): Proportion of flux provided by the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
		delta_inc (float): Distance between inclination guesses.
		inc_guess (float): Initial inclination guess.
		delta_R_p (float): Distance between planet radius guesses.
		R_p_guess (float): Initial planet radius guess.
	Returns:
		lnLs[best_idx] (float): Log likelihood of best estimate scenario.
		incs[best_idx[0]] (float): Inclination [degrees] of best estimate scenario.
		R_p[best_idx[1]] (float): Planet radius [Earth radii] of best estimate scenario.
		delta_inc (float): New distance between inclination guesses.
		delta_R_p (float): New distance between planet radius guesses.
	"""

	if inc_guess == None and R_p_guess == None:
		lnLs = np.zeros([N,M])
		incs = np.linspace(inc_min,inc_max,N)
		delta_inc = incs[1]-incs[0]
		R_p = np.linspace(R_p_min,R_p_max,M)
		delta_R_p = R_p[1]-R_p[0]
		for j in range(N):
			for k in range(M):
				lnLs[j,k] = log_likelihood_TP([incs[j],R_p[k]], time, flux, flux_err, P_orb, a, R_s, u1, u2, companion_fluxratio, companion)
		best_idx = np.unravel_index(lnLs.argmin(), lnLs.shape)
		return lnLs[best_idx], incs[best_idx[0]], R_p[best_idx[1]], delta_inc, delta_R_p
	else:
		lnLs = np.zeros([N,M])
		incs = np.linspace(np.max([inc_guess-5*delta_inc, inc_min]),np.min([inc_guess+5*delta_inc, inc_max]),N)
		delta_inc = incs[1]-incs[0]
		R_p = np.linspace(np.max([R_p_guess-5*delta_R_p, R_p_min]),np.min([R_p_guess+5*delta_R_p, R_p_max]),M)
		delta_R_p = R_p[1]-R_p[0]
		for j in range(N):
			for k in range(M):
				lnLs[j,k] = log_likelihood_TP([incs[j],R_p[k]], time, flux, flux_err, P_orb, a, R_s, u1, u2, companion_fluxratio, companion)
		best_idx = np.unravel_index(lnLs.argmin(), lnLs.shape)
		return lnLs[best_idx], incs[best_idx[0]], R_p[best_idx[1]], delta_inc, delta_R_p
	
def iterate_grid_minimize_TP(time, flux, flux_err, P_orb:float, a:float, R_s:float, u1:float, u2:float, niter:int, N:int, M:int, inc_min:float, inc_max:float, R_p_min:float, R_p_max:float, companion_fluxratio:float = 0.0, companion:bool = False):
	"""
	Iteratively searches for most likely transiting planet scenario (used to get a good initial guess).
	Args:
		x0 (numpy array): Inclindation [degrees] and planet radius [Earth radii].
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		a (float): Semimajor axis [cm].
		R_s (float): Star radius [Solar radii].
		u1 (float): 1st coefficient in quadratic limb darkening law.
		u2 (float): 2nd coefficient in quadratic limb darkening law.
		niter (int): Number of iterations.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 
		inc_min (float): Minimum inclination considered.
		inc_max (float): Maximum inclination considered.
		R_p_min (float): Minimum planet radius considered.
		R_p_max (float): Maximum planet radius considered.
		companion_fluxratio (float): Proportion of flux provided by the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
	Returns: 
		lnL (float): Log likelihood of best guess.
		inc_guess (float): Best guess inclination [degrees].
		R_p_guess (float): Best guess planet radius [Earth radii].
	"""

	inc_guess, R_p_guess = None, None
	delta_inc, delta_R_p = None, None
	for i in range(niter):
		lnL, inc_guess, R_p_guess, delta_inc, delta_R_p = grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, N, M, inc_min, inc_max, R_p_min, R_p_max, companion_fluxratio, companion, delta_inc, inc_guess, delta_R_p, R_p_guess)
	return lnL, inc_guess, R_p_guess

def grid_minimize_EB(time, flux, flux_err, P_orb, lum, N, M, inc_min, inc_max, fr_min, fr_max, companion_fluxratio:float = 0.0, companion:bool = False, delta_inc = None, inc_guess = None, delta_fr = None, fr_guess = None):
	"""
	Makes an initial guess at the most likely transiting planet scenario using a grid of inclinations and planet radii.
	Args:
		x0 (numpy array): Inclindation [degrees] and EB flux ratio.
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 
		inc_min (float): Minimum inclination considered.
		inc_max (float): Maximum inclination considered.
		fr_min (float): Minimum EB flux ratio considered.
		fr_max (float): Maximum EB flux ratio considered.
		companion_fluxratio (float): Proportion of flux provided by the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
		delta_inc (float): Distance between inclination guesses.
		inc_guess (float): Initial inclination guess.
		delta_fr (float): Distance between EB flux ratio guesses.
		fr_guess (float): Initial EB flux ratio guess.
	Returns:
		lnLs[best_idx]: Log likelihood of best estimate scenario.
		incs[best_idx[0]]: Inclination [degrees] of best estimate scenario.
		fr[best_idx[1]]: EB flux ratio of best estimate scenario.
		delta_inc: New distance between inclination guesses.
		delta_fr: New distance between EB flux ratio guesses.
	"""

	if inc_guess == None and fr_guess == None:
		lnLs = np.zeros([N,M])
		incs = np.linspace(inc_min,inc_max,N)
		delta_inc = incs[1]-incs[0]
		# fr = np.linspace(fr_min,fr_max,M)
		fr = np.logspace(np.log10(fr_min),np.log10(fr_max),M)
		delta_fr = fr[1]-fr[0]
		for j in range(N):
			for k in range(M):
				lnLs[j,k] = log_likelihood_EB([incs[j],fr[k]], time, flux, flux_err, P_orb, lum, companion_fluxratio, companion)
		best_idx = np.unravel_index(lnLs.argmin(), lnLs.shape)
		return lnLs[best_idx], incs[best_idx[0]], fr[best_idx[1]], delta_inc, delta_fr
	else:
		lnLs = np.zeros([N,M])
		incs = np.linspace(np.max([inc_guess-5*delta_inc, inc_min]),np.min([inc_guess+5*delta_inc, inc_max]),N)
		delta_inc = incs[1]-incs[0]
		fr = np.linspace(np.max([fr_guess-5*delta_fr, fr_min]),np.min([fr_guess+5*delta_fr, fr_max]),M)
		delta_fr = fr[1]-fr[0]
		for j in range(N):
			for k in range(M):
				lnLs[j,k] = log_likelihood_EB([incs[j],fr[k]], time, flux, flux_err, P_orb, lum, companion_fluxratio, companion)
		best_idx = np.unravel_index(lnLs.argmin(), lnLs.shape)
		return lnLs[best_idx], incs[best_idx[0]], fr[best_idx[1]], delta_inc, delta_fr
	
def iterate_grid_minimize_EB(time, flux, flux_err, P_orb:float, lum:float, niter:int, N:int, M:int, inc_min:float, inc_max:float, fr_min:float, fr_max:float, companion_fluxratio:float = 0.0, companion:bool = False):
	"""
	Iteratively searches for most likely eclipsing binary scenario (used to get a good initial guess).
	Args:
		x0 (numpy array): Inclindation [degrees] and EB flux ratio..
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		lum (float): Star luminosity [Solar luminosities].
		niter (int): Number of iterations.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 
		inc_min (float): Minimum inclination considered.
		inc_max (float): Maximum inclination considered.
		fr_min (float): Minimum EB flux ratio considered.
		fr_max (float): Maximum EB flux ratio considered.
		companion_fluxratio (float): Proportion of flux provided by the unresolved companion.
		companion (bool): True if the transit is around the unresolved companion and False if it is not. 
	Returns: 
		lnL (float): Log likelihood of best guess.
		inc_guess (float): Best guess inclination [degrees].
		fr_guess (float): Best guess EB flux ratio.
	"""

	inc_guess, fr_guess = None, None
	delta_inc, delta_fr = None, None
	for i in range(niter):
		lnL, inc_guess, fr_guess, delta_inc, delta_fr = grid_minimize_EB(time, flux, flux_err, P_orb, lum, N, M, inc_min, inc_max, fr_min, fr_max, companion_fluxratio, companion, delta_inc, inc_guess, delta_fr, fr_guess)
	return lnL, inc_guess, fr_guess

def likelihood_TTP(time, flux, flux_err, P_orb:float, M_s:float, R_s:float, Teff:float, logg:float, Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the TTP scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		R_s (float): Star mass [Solar masses].
		R_s (float): Star radius [Solar radii].
		Teff (float): Star effective temperature [K].
		logg (float): Star logg.
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely planet radius [Earth radii].
	"""

	a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
	inc_max = 90.0
	inc_min = np.arccos(2*R_s*constants.R_sun.cgs.value/a) * 180/np.pi
	R_p_max = 32.5
	R_p_min = 0.5
	u1, u2 = quad_coeffs(Teff, logg, Z)

	lnL_guess, inc_guess, R_p_guess = iterate_grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, niter, N, M, inc_min, inc_max, R_p_min, R_p_max, 0.0, False)
	results = minimize(log_likelihood_TP, [inc_guess, R_p_guess], args=(time, flux, flux_err, P_orb, a, R_s, u1, u2, 0.0, False), method="Nelder-Mead")

	return results.fun, results.x[0], results.x[1]

def likelihood_TEB(time, flux, flux_err, P_orb:float, lum:float, Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the TEB scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of EB flux ratio guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely EB flux ratio.
	"""
	
	EB_fluxratio_max = 0.5
	EB_fluxratio_min = np.min([0.001/lum, 0.5])
	inc_max = 90.0
	inc_min = 0.0
	lnL_guess, inc_guess, EB_fluxratio_guess = iterate_grid_minimize_EB(time, flux, flux_err, P_orb, lum, niter, N, M, inc_min, inc_max, EB_fluxratio_min, EB_fluxratio_max, 0.0, False)
	results = minimize(log_likelihood_EB, [inc_guess, EB_fluxratio_guess], args=(time, flux, flux_err, P_orb, lum, 0.0, False), method="Nelder-Mead")

	return results.fun, results.x[0], results.x[1]

def likelihood_PTP(time, flux, flux_err, P_orb:float, lum:float,  Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the PTP scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely planet radius [Earth radii].
		fluxratio (float): Most likely proportion of flux from unresolved companion.
	"""
	
	fluxratio_min = 0.001/lum
	Delta_mag_min = 2.5*np.log10(fluxratio_min)
	Delta_mags = np.arange( np.max([np.ceil(Delta_mag_min), -8]), 1, 1 )
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([fluxratio_array.shape[0], 4])
	for i, fluxratio in enumerate(fluxratio_array):
		new_lum = lum*(1-fluxratio)
		M_s, R_s, Teff, logg, new_lum = stellar_relations(lum=new_lum)
		a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
		inc_max = 90.0
		inc_min = np.arccos(2*R_s*constants.R_sun.cgs.value/a) * 180/np.pi
		R_p_max = 32.5
		R_p_min = 0.5
		u1, u2 = quad_coeffs(Teff, logg, Z)
		lnL_guess, inc_guess, R_p_guess = iterate_grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, niter, N, M, inc_min, inc_max, R_p_min, R_p_max, fluxratio, False)
		results = minimize(log_likelihood_TP, [inc_guess, R_p_guess], args=(time, flux, flux_err, P_orb, a, R_s, u1, u2, fluxratio, False), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio])

	best_idx = all_results[:,0].argmin()
	return all_results[best_idx]

def likelihood_DTP(time, flux, flux_err, P_orb:float, lum:float,  Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the DTP scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely planet radius [Earth radii].
		fluxratio (float): Most likely proportion of flux from unresolved companion.
	"""
	
	Delta_mags = np.arange(-8, 1, 1)
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([fluxratio_array.shape[0], 4])
	for i, fluxratio in enumerate(fluxratio_array):
		new_lum = lum*(1-fluxratio)
		M_s, R_s, Teff, logg, new_lum = stellar_relations(lum=new_lum)
		a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
		inc_max = 90.0
		inc_min = np.arccos(2*R_s*constants.R_sun.cgs.value/a) * 180/np.pi
		R_p_max = 32.5
		R_p_min = 0.5
		u1, u2 = quad_coeffs(Teff, logg, Z)
		lnL_guess, inc_guess, R_p_guess = iterate_grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, niter, N, M, inc_min, inc_max, R_p_min, R_p_max, fluxratio, False)
		results = minimize(log_likelihood_TP, [inc_guess, R_p_guess], args=(time, flux, flux_err, P_orb, a, R_s, u1, u2, fluxratio, False), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio])

	best_idx = all_results[:,0].argmin()
	return all_results[best_idx]

def likelihood_PEB(time, flux, flux_err, P_orb:float, lum:float,  Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the PTP scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of EB flux ratio guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely EB flux ratio.
		fluxratio (float): Most likely proportion of flux from unresolved companion.
	"""
	
	fluxratio_min = 0.001/lum
	Delta_mag_min = 2.5*np.log10(fluxratio_min)
	Delta_mags = np.arange( np.max([np.ceil(Delta_mag_min), -4]), 1, 1 )
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([fluxratio_array.shape[0], 4])
	for i, fluxratio in enumerate(fluxratio_array):
		new_lum = lum*(1-fluxratio)
		EB_fluxratio_max = 0.5
		EB_fluxratio_min = np.min([0.001/lum, 0.5])
		inc_max = 90.0
		inc_min = 0.0
		lnL_guess, inc_guess, EB_fluxratio_guess = iterate_grid_minimize_EB(time, flux, flux_err, P_orb, new_lum, niter, N, M, inc_min, inc_max, EB_fluxratio_min, EB_fluxratio_max, fluxratio, False)
		results = minimize(log_likelihood_EB, [inc_guess, EB_fluxratio_guess], args=(time, flux, flux_err, P_orb, new_lum, fluxratio, False), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio])

	best_idx = all_results[:,0].argmin()
	return all_results[best_idx]

def likelihood_DEB(time, flux, flux_err, P_orb:float, lum:float,  Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the DEB scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of EB flux ratio guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely EB flux ratio.
		fluxratio (float): Most likely proportion of flux from unresolved companion.
	"""
	
	Delta_mags = np.arange(-4, 1, 1)
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([fluxratio_array.shape[0], 4])
	for i, fluxratio in enumerate(fluxratio_array):
		new_lum = lum*(1-fluxratio)
		EB_fluxratio_max = 0.5
		EB_fluxratio_min = np.min([0.001/lum, 0.5])
		inc_max = 90.0
		inc_min = 0.0
		lnL_guess, inc_guess, EB_fluxratio_guess = iterate_grid_minimize_EB(time, flux, flux_err, P_orb, new_lum, niter, N, M, inc_min, inc_max, EB_fluxratio_min, EB_fluxratio_max, fluxratio, False)
		results = minimize(log_likelihood_EB, [inc_guess, EB_fluxratio_guess], args=(time, flux, flux_err, P_orb, new_lum, fluxratio, False), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio])

	best_idx = all_results[:,0].argmin()
	return all_results[best_idx]

def likelihood_STP(time, flux, flux_err, P_orb:float, lum:float,  Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the STP scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely planet radius [Earth radii].
		fluxratio (float): Most likely proportion of flux from unresolved companion.
	"""

	fluxratio_min = 0.001/lum
	Delta_mag_min = 2.5*np.log10(fluxratio_min)
	Delta_mags = np.arange( np.max([np.ceil(Delta_mag_min), -8]), 1, 1 )
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([fluxratio_array.shape[0], 4])
	for i, fluxratio in enumerate(fluxratio_array):
		new_lum = lum*fluxratio
		M_s, R_s, Teff, logg, new_lum = stellar_relations(lum=new_lum)
		a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
		inc_max = 90.0
		inc_min = np.arccos(2*R_s*constants.R_sun.cgs.value/a) * 180/np.pi
		R_p_max = 32.5
		R_p_min = 0.5
		u1, u2 = quad_coeffs(Teff, logg, Z)
		lnL_guess, inc_guess, R_p_guess = iterate_grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, niter, N, M, inc_min, inc_max, R_p_min, R_p_max, fluxratio, True)
		results = minimize(log_likelihood_TP, [inc_guess, R_p_guess], args=(time, flux, flux_err, P_orb, a, R_s, u1, u2, fluxratio, True), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio])

	best_idx = all_results[:,0].argmin()
	return all_results[best_idx]

def likelihood_SEB(time, flux, flux_err, P_orb:float, lum:float,  Z:float = 0.0, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the SEB scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		lum (float): Star luminosity [Solar luminosities].
		Z (float): Star metallicity.
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of EB flux ratio guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely EB flux ratio.
		fluxratio (float): Most likely proportion of flux from unresolved companion.
	"""
	
	fluxratio_min = 0.001/lum
	Delta_mag_min = 2.5*np.log10(fluxratio_min)
	Delta_mags = np.arange( np.max([np.ceil(Delta_mag_min), -8]), 1, 1 )
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([fluxratio_array.shape[0], 4])
	for i, fluxratio in enumerate(fluxratio_array):
		new_lum = lum*fluxratio
		EB_fluxratio_max = 0.5
		EB_fluxratio_min = np.min([0.001/lum, 0.5])
		inc_max = 90.0
		inc_min = 0.0
		lnL_guess, inc_guess, EB_fluxratio_guess = iterate_grid_minimize_EB(time, flux, flux_err, P_orb, new_lum, niter, N, M, inc_min, inc_max, EB_fluxratio_min, EB_fluxratio_max, fluxratio, True)
		results = minimize(log_likelihood_EB, [inc_guess, EB_fluxratio_guess], args=(time, flux, flux_err, P_orb, new_lum, fluxratio, True), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio])

	best_idx = all_results[:,0].argmin()
	return all_results[best_idx]

def likelihood_BTP(time, flux, flux_err, P_orb:float, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the BTP scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely planet radius [Earth radii].
		fluxratio (float): Most likely proportion of flux from unresolved companion.
		Teff (float): Most likely effective temperature [K] of unresolved companion.
	"""

	# Teff_array = np.array([3500,4500,5500,7000,9000,15000,40000])
	Teff_array = np.array([4000,6000,8000,10000])
	Delta_mags = np.arange(-8, 1, 1)
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([Teff_array.shape[0], fluxratio_array.shape[0], 5])
	for i, Teff_i in enumerate(Teff_array):
		# properties of fg/bg star
		M_s, R_s, Teff, logg, lum = stellar_relations(Teff=Teff_i)
		a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
		inc_max = 90.0
		inc_min = np.arccos(2*R_s*constants.R_sun.cgs.value/a) * 180/np.pi
		R_p_max = 32.5
		R_p_min = 0.5
		u1, u2 = quad_coeffs(Teff, logg, 0.0)
		for j, fluxratio in enumerate(fluxratio_array):
			lnL_guess, inc_guess, R_p_guess = iterate_grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, niter, N, M, inc_min, inc_max, R_p_min, R_p_max, fluxratio, True)
			results = minimize(log_likelihood_TP, [inc_guess, R_p_guess], args=(time, flux, flux_err, P_orb, a, R_s, u1, u2, fluxratio, True), method="Nelder-Mead")
			all_results[i,j,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio, Teff])

	best_idx = np.unravel_index(all_results[:,:,0].argmin(), all_results[:,:,0].shape)
	return all_results[best_idx]

def likelihood_BEB(time, flux, flux_err, P_orb:float, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the BEB scenario.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of EB flux ratio guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely EB flux ratio.
		fluxratio (float): Most likely proportion of flux from unresolved companion.
		Teff (float): Most likely effective temperature [K] of unresolved companion.
	"""
	
	# Teff_array = np.array([3500,4500,5500,7000,9000,15000,40000])
	Teff_array = np.array([4000,6000,8000,10000])
	Delta_mags = np.arange(-8, 1, 1)
	fluxratio_array = 10**(Delta_mags/2.5) / (1 + 10**(Delta_mags/2.5))
	all_results = np.zeros([Teff_array.shape[0], fluxratio_array.shape[0], 5])
	for i, Teff_i in enumerate(Teff_array):
		M_s, R_s, Teff, logg, lum = stellar_relations(Teff=Teff_i)
		EB_fluxratio_max = 0.5
		EB_fluxratio_min = np.min([0.001/lum, 0.5])
		inc_max = 90.0
		inc_min = 0.0
		for j, fluxratio in enumerate(fluxratio_array): 
			lnL_guess, inc_guess, EB_fluxratio_guess = iterate_grid_minimize_EB(time, flux, flux_err, P_orb, lum, niter, N, M, inc_min, inc_max, EB_fluxratio_min, EB_fluxratio_max, fluxratio, True)
			results = minimize(log_likelihood_EB, [inc_guess, EB_fluxratio_guess], args=(time, flux, flux_err, P_orb, lum, fluxratio, True), method="Nelder-Mead")
			all_results[i,j,:] = np.array([results.fun, results.x[0], results.x[1], fluxratio, Teff])

	best_idx = np.unravel_index(all_results[:,:,0].argmin(), all_results[:,:,0].shape)
	return all_results[best_idx]


def likelihood_uncharacterized_NTP(time, flux, flux_err, P_orb:float, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the NTP scenario when the star is uncharacterized.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of inclination guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely planet radius [Earth radii].
		Teff (float): Most likely effective temperature [K] of unresolved companion.
	"""

	# Teff_array = np.array([3500,4500,5500,7000,9000,15000,40000])
	Teff_array = np.array([4000,6000,8000,10000])
	all_results = np.zeros([Teff_array.shape[0], 4])
	for i, Teff_i in enumerate(Teff_array):
		# properties of fg/bg star
		M_s, R_s, Teff, logg, lum = stellar_relations(Teff=Teff_i)
		a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
		inc_max = 90.0
		inc_min = np.arccos(2*R_s*constants.R_sun.cgs.value/a) * 180/np.pi
		R_p_max = 32.5
		R_p_min = 0.5
		u1, u2 = quad_coeffs(Teff, logg, 0.0)
		lnL_guess, inc_guess, R_p_guess = iterate_grid_minimize_TP(time, flux, flux_err, P_orb, a, R_s, u1, u2, niter, N, M, inc_min, inc_max, R_p_min, R_p_max, 0.0, False)
		results = minimize(log_likelihood_TP, [inc_guess, R_p_guess], args=(time, flux, flux_err, P_orb, a, R_s, u1, u2, 0.0, False), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], Teff])

	best_idx = np.unravel_index(all_results[:,0].argmin(), all_results[:,0].shape)
	return all_results[best_idx]

def likelihood_uncharacterized_NEB(time, flux, flux_err, P_orb:float, niter:int = 1, N:int = 30, M:int = 30):
	"""
	Maximizes the log likelihood for the NEB scenario when the star is uncharacterized.
	Args:
		time (numpy array): Time of each data point [days from transit midpoint].
		flux (numpy array): Normalized flux of each data point.
		flux_err (numpy array): Normalized flux error of each data point.
		P_orb (float): Orbital period [days].
		niter (int): Number of iterations when searching for initial guesses.
		N (int): Resolution of EB flux ratio guesses.
		M (int): Resolution of planet radius guesses. 	
	Returns:
		results.fun (float): Maximized log likelihood
		results.x[0] (float): Most likely inclination [deg].
		results.x[1] (float): Most likely EB flux ratio.
		Teff (float): Most likely effective temperature [K] of unresolved companion.
	"""
	
	# Teff_array = np.array([3500,4500,5500,7000,9000,15000,40000])
	Teff_array = np.array([4000,6000,8000,10000])
	all_results = np.zeros([Teff_array.shape[0], 4])
	for i, Teff_i in enumerate(Teff_array):
		M_s, R_s, Teff, logg, lum = stellar_relations(Teff=Teff_i)
		EB_fluxratio_max = 0.5
		EB_fluxratio_min = np.min([0.001/lum, 0.5])
		inc_max = 90.0
		inc_min = 0.0
		lnL_guess, inc_guess, EB_fluxratio_guess = iterate_grid_minimize_EB(time, flux, flux_err, P_orb, lum, niter, N, M, inc_min, inc_max, EB_fluxratio_min, EB_fluxratio_max, 0.0, False)
		results = minimize(log_likelihood_EB, [inc_guess, EB_fluxratio_guess], args=(time, flux, flux_err, P_orb, lum, 0.0, False), method="Nelder-Mead")
		all_results[i,:] = np.array([results.fun, results.x[0], results.x[1], Teff])

	best_idx = np.unravel_index(all_results[:,0].argmin(), all_results[:,0].shape)
	return all_results[best_idx]