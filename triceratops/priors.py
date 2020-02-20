import numpy as np
from pandas import read_csv
from scipy.stats import gaussian_kde
from scipy.integrate import quad, simps
from astropy import constants
from mechanicalsoup import StatefulBrowser
from time import sleep

def f_companion(M_s:float, max_sep:float):
	"""
	Calculates the bound companion rate of the target star.
	Args:
		M_s (float): Star mass [Solar masses].
		max_sep (float): Maximum physical separation resolved [au]. 
	Returns:
		f_act (float): Companion rate of target star/
	"""

	if M_s >= 1.0:
		f1 = 0.020 + 0.04*np.log10(M_s) + 0.07*(np.log10(M_s))**2
		f2 = 0.039 + 0.07*np.log10(M_s) + 0.01*(np.log10(M_s))**2
		f3 = 0.078 - 0.05*np.log10(M_s) + 0.04*(np.log10(M_s))**2
		alpha = 0.018
		dlogP = 0.7
		max_Porb = ((4*np.pi**2)/(constants.G.cgs.value*M_s*constants.M_sun.cgs.value)*(max_sep*constants.au.cgs.value)**3)**(1/2) / 86400
		if np.log10(max_Porb) < 1.0:
			t1 = f1
			f_comp = t1
		elif 1.0 <= np.log10(max_Porb) < 2.0:
			t1 = f1
			t2 = 0.5*(np.log10(max_Porb) - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(np.log10(max_Porb) - 1.0))
			f_comp = t1 + t2
		elif 2.0 <= np.log10(max_Porb) < 3.4:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(np.log10(max_Porb)**2 - 5.4*np.log10(max_Porb) + 6.8) + f2*(np.log10(max_Porb) - 2.0)
			f_comp = t1 + t2 + t3 
		elif 3.4 <= np.log10(max_Porb) < 5.5:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
			t4 = alpha*dlogP*(np.log10(max_Porb) - 3.4) + f2*(np.log10(max_Porb) - 3.4) + (f3 - f2 - alpha*dlogP)*(0.238095*np.log10(max_Porb)**2 - 0.952381*np.log10(max_Porb) + 0.485714)
			f_comp = t1 + t2 + t3 + t4
		elif 5.5 <= np.log10(max_Porb) < 8.0:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
			t4 = alpha*dlogP*(5.5 - 3.4) + f2*(5.5 - 3.4) + (f3 - f2 - alpha*dlogP)*(0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
			t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*np.log10(max_Porb)))
			f_comp = t1 + t2 + t3 + t4 + t5
		elif np.log10(max_Porb) >= 8.0:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
			t4 = alpha*dlogP*(5.5 - 3.4) + f2*(5.5 - 3.4) + (f3 - f2 - alpha*dlogP)*(0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
			t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*8.0))
			f_comp = t1 + t2 + t3 + t4 + t5  
		return f_comp
	else:
		M_act = M_s
		M_s = 1.0
		f1 = 0.020 + 0.04*np.log10(M_s) + 0.07*(np.log10(M_s))**2
		f2 = 0.039 + 0.07*np.log10(M_s) + 0.01*(np.log10(M_s))**2
		f3 = 0.078 - 0.05*np.log10(M_s) + 0.04*(np.log10(M_s))**2
		alpha = 0.018
		dlogP = 0.7
		max_Porb = ((4*np.pi**2)/(constants.G.cgs.value*M_s*constants.M_sun.cgs.value)*(max_sep*constants.au.cgs.value)**3)**(1/2) / 86400
		if np.log10(max_Porb) < 1.0:
			t1 = f1
			f_comp = t1
		elif 1.0 <= np.log10(max_Porb) < 2.0:
			t1 = f1
			t2 = 0.5*(np.log10(max_Porb) - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(np.log10(max_Porb) - 1.0))
			f_comp = t1 + t2
		elif 2.0 <= np.log10(max_Porb) < 3.4:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(np.log10(max_Porb)**2 - 5.4*np.log10(max_Porb) + 6.8) + f2*(np.log10(max_Porb) - 2.0)
			f_comp = t1 + t2 + t3 
		elif 3.4 <= np.log10(max_Porb) < 5.5:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
			t4 = alpha*dlogP*(np.log10(max_Porb) - 3.4) + f2*(np.log10(max_Porb) - 3.4) + (f3 - f2 - alpha*dlogP)*(0.238095*np.log10(max_Porb)**2 - 0.952381*np.log10(max_Porb) + 0.485714)
			f_comp = t1 + t2 + t3 + t4
		elif 5.5 <= np.log10(max_Porb) < 8.0:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
			t4 = alpha*dlogP*(5.5 - 3.4) + f2*(5.5 - 3.4) + (f3 - f2 - alpha*dlogP)*(0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
			t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*np.log10(max_Porb)))
			f_comp = t1 + t2 + t3 + t4 + t5
		elif np.log10(max_Porb) >= 8.0:
			t1 = f1
			t2 = 0.5*(2.0 - 1.0)*(2.0*f1 + (f2 - f1 - alpha*dlogP)*(2.0 - 1.0))
			t3 = 0.5*alpha*(3.4**2 - 5.4*3.4 + 6.8) + f2*(3.4 - 2.0)
			t4 = alpha*dlogP*(5.5 - 3.4) + f2*(5.5 - 3.4) + (f3 - f2 - alpha*dlogP)*(0.238095*5.5**2 - 0.952381*5.5 + 0.485714)
			t5 = f3*(3.33333 - 17.3566*np.exp(-0.3*8.0))
			f_comp = t1 + t2 + t3 + t4 + t5  
		f_act =  0.65*f_comp+0.35*f_comp*M_act
		return f_act

def rate_planet(M_s:float):
	"""
	Estimates planet occurrence rate for all planet radii and orbital periods under 50 days.
	Args:
		M_s (float): Star mass [Solar masses].
	Returns:
		Planet occurrence rate.
	"""

	if (2.5 - 1.5*M_s) > 0.1:
		return 2.5 - 1.5*M_s
	else:
		return 0.1

def rate_binary(M_s:float):
	"""
	Estimates bound companion rate for binaries with orbital periods under 50 days.
	Args:
		M_s (float): Star mass [Solar masses].
	Returns:
		Companion rate.
	"""

	max_sep = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(50*86400)**2)**(1/3) / constants.au.cgs.value
	return f_companion(M_s, max_sep)

def prob_TP_Porb(P_orb:float):
	"""
	Calculates probability of a planet with a given orbital period.
	Args:
		P_orb (float): Orbital period [days].
	Returns:
		Probability.
	"""

	if P_orb < 0.2:
		P_orb = 0.2
	elif P_orb > (10**(np.log10(50) - 0.1)):
		P_orb = 10**(np.log10(50) - 0.1)
	prob = lambda x: np.log10(10**x + 0.8)
	if (np.log10(P_orb)-0.1) < np.log10(0.2):
		return quad(prob, np.log10(0.2), np.log10(P_orb)+0.1)[0] / quad(prob, np.log10(0.2), np.log10(50))[0]
	elif (np.log10(P_orb)+0.1) > np.log10(50):
		return quad(prob, np.log10(P_orb)-0.1, np.log10(50))[0] / quad(prob, np.log10(0.2), np.log10(50))[0]
	else:
		return quad(prob, np.log10(P_orb)-0.1, np.log10(P_orb)+0.1)[0] / quad(prob, np.log10(0.2), np.log10(50))[0]

def prob_TP_Rp(R_p:float, M_s:float):
	"""
	Calculates probability of a planet with a given radius.
	Args:
		R_p (float): Planet radius [Earth radii].
		M_s (float): Star mass [Solar masses].
	Returns:
		Probability.
	"""

	if M_s <= 0.45:
		prob = lambda x: 10**(2/(1+np.exp(1.5*(x-4))) - x/10)
	else:
		prob = lambda x: 10**(1/(1+np.exp(1.5*(x-4))) - x/10)
	if (R_p <= 0) or (R_p >= 33):
		return 0
	elif(R_p > 0) and (R_p < 0.5):
		return quad(prob, 0, 1)[0] / quad(prob, 0, 20)[0]
	elif (R_p > 19.5) and (R_p < 33):
		return quad(prob, 19, 20)[0] / quad(prob, 0, 20)[0]
	else:
		return quad(prob, R_p-0.5, R_p+0.5)[0] / quad(prob, 0, 20)[0]

def prob_TP_geo(P_orb:float, tdepth:float, M_s:float, R_s:float):
	"""
	Calculates geometric transit probability for transiting planets.
	Args:
		R_p (float): Planet radius [Earth radii].
		tdepth (float): Reported transit depth [ppm].
		M_s (float): Star mass [Solar masses].
		R_s (float): Star radius [Solar radii].
	Returns:
		Geometric transit probability.
	"""

	a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
	return R_s*constants.R_sun.cgs.value*(1+np.sqrt(tdepth))/a

def prob_EB_Porb(P_orb:float, tdepth:float, M_s:float, R_s:float, EB_periods):
	"""
	Calculates probability of an EB with a given orbital periiod.
	Args:
		P_orb (float): Orbital period [days].
		tdepth (float): Reported transit depth [ppm].
		M_s (float): Star mass [Solar masses].
		R_s (float): Star radius [Solar radii].
		EB_periods (list): List of EB periods from the KEBC.
	Returns:
		Geometric transit probability.
	"""

	if P_orb < 0.2:
		P_orb = 0.2
	elif P_orb > (10**(np.log10(50) - 0.1)):
		P_orb = 10**(np.log10(50) - 0.1)
	hist = np.histogram(np.log10(EB_periods), bins=np.linspace(np.log10(0.2),np.log10(50),20))
	dataset = []
	for i in range(len(hist[0])):
		for j in range(int(hist[0][i])):
			dataset.append(hist[1][i])
	kde = gaussian_kde(dataset, 0.2)
	x = np.linspace(np.log10(0.2),np.log10(50),10000)
	geo_corr = R_s*constants.R_sun.cgs.value*(1+np.sqrt(tdepth))/(((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(10**x * 86400)**2)**(1/3))
	prob = kde.pdf(x)/geo_corr
	idx_lower = np.argmin(abs(x-(np.log10(P_orb)-0.1)))
	idx_upper = np.argmin(abs(x-(np.log10(P_orb)+0.1)))
	return simps(prob[idx_lower+1:idx_upper], x[idx_lower+1:idx_upper]) / simps(prob, x)

def prob_EB_Rratio(tdepth:float, R_s:float, EB_pdepths):
	"""
	Calculates probability of an EB of a given size.
	Args:
		tdepth (float): Reported transit depth [ppm].
		R_s (float): Star radius [Solar radii].
		EB_pdepths [list]: List of EB transit depths from the KEBC.
	Returns:
		Probability.
	"""

	min_pdepth = (0.1/R_s)**2
	int_limit = (1-np.sqrt(min_pdepth))/20/2
	hist = np.histogram(np.sqrt(EB_pdepths[EB_pdepths >= min_pdepth]), bins=np.linspace(np.sqrt(min_pdepth),1,20))
	dataset = []
	for i in range(len(hist[0])):
		for j in range(int(hist[0][i])):
			dataset.append(hist[1][i])  
	if len(dataset) > 0:
		kde = gaussian_kde(dataset, 0.2)
		x = np.linspace(np.sqrt(min_pdepth),1,10000)
		prob = kde.pdf(x)
		idx_lower = np.argmin(abs(x-(np.sqrt(tdepth)-int_limit)))
		idx_upper = np.argmin(abs(x-(np.sqrt(tdepth)+int_limit)))
		return simps(prob[idx_lower+1:idx_upper], x[idx_lower+1:idx_upper]) / simps(prob, x)
	else:
		return 1/40
	
def prob_EB_geo(P_orb:float, tdepth:float, M_s:float, R_s:float):
	"""
	Calculates geometric transit probability for eclipsing binaries.
	Args:
		P_orb (float): Orbital period [days].
		tdepth (float): Reported transit depth [ppm].
		M_s (float): Star mass [Solar masses].
		R_s (float): Star radius [Solar radii].
	Returns:
		Geometric transit probability.
	"""

	a = ((constants.G.cgs.value*M_s*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
	return R_s*constants.R_sun.cgs.value*(1+np.sqrt(tdepth))/a
	
def prior_TP(P_orb:float, tdepth:float, M_s:float, R_s:float):
	"""of
	Calculates the prior probability of a transiting planet scenario.
	Args:
		P_orb (float): Orbital period [days].
		tdepth (float): Reported transit depth [ppm].
		M_s (float): Star mass [Solar masses].
		R_s (float): Star radius [Solar radii].
	Returns:
		Prior probability of TP scenario. 
	"""

	R_p = np.sqrt(tdepth)*R_s*constants.R_sun.cgs.value/constants.R_earth.cgs.value
	return rate_planet(M_s) * prob_TP_Porb(P_orb) * prob_TP_Rp(R_p, M_s) * prob_TP_geo(P_orb, tdepth, M_s, R_s)
	
def prior_EB(P_orb:float, tdepth:float, M_s:float, R_s:float, EB_periods, EB_pdepths):
	"""
	Calculates the prior probability of an eclipsing binary scenario.
	Args:
		P_orb (float): Orbital period [days].
		tdepth (float): Reported transit depth [ppm].
		M_s (float): Star mass [Solar masses].
		R_s (float): Star radius [Solar radii].
		EB_periods (list): List of EB periods from the KEBC.
		EB_pdepths [list]: List of EB transit depths from the KEBC.
	Returns:
		Prior probability of an EB scenario.
	"""

	return rate_binary(M_s) * prob_EB_Porb(P_orb, tdepth, M_s, R_s, EB_periods) * prob_EB_Rratio(tdepth, R_s, EB_pdepths) * prob_EB_geo(P_orb, tdepth, M_s, R_s)

def prior_bound_companion(M_s:float, plx:float, best_resolution:float = 2.2):
	"""
	Calculates the prior probability of a bound companion.
	Args:
		M_s (float): Star mass [Solar masses].
		plx (float): Star parallax [mas].
		best_resolution (float): Best angular resolution beyond which a companion with a given magnitude can be outruled.
	Returns:
		f_companion (float): Bound companion rate.
	"""

	if np.isnan(plx) == True:
		plx = 0.1
	d = 1000/plx
	max_sep = best_resolution*d
	return f_companion(M_s, max_sep)

def query_TRILEGAL(RA:float, Dec:float):
	"""
	Begins TRILEGAL query.
	Args:
		RA, Dec: Coordinates of the target.
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
	print("TRILEGAL form submitted.")
	sleep(5)
	if len(browser.get_current_page().select("a")) == 0:
		# print("TRILEGAL too busy, using saved stellar populations instead.")
		# return None
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
			print("TRILEGAL too busy, using saved stellar populations instead.")
			return None		
		else:
			data_link = browser.get_current_page().select("a")[0].get("href")
			output_url = "http://stev.oapd.inaf.it/"+data_link[3:]
			return output_url	
	else:
		data_link = browser.get_current_page().select("a")[0].get("href")
		output_url = "http://stev.oapd.inaf.it/"+data_link[3:]
		return output_url

def prior_unbound_companion(output_url, Tmag:float, best_resolution:float = 2.2):
	"""
	Calculates the prior probability of an unbound companion.
	Args:
		output_url (str): URL of page with query results. 
		Tmag (float): TESS magnitude of the star.
		best_resolution (float): Best angular resolution beyond which a companion with a given magnitude can be outruled.
	Returns:
		f_fgbg (float): Unbound companion rate.
	"""

	if output_url == None:
		print("Could not access TRILEGAL. Ignoring BTP, BEB, DTP, and DEB scenarios.")
		return 0.0
	else:
		for i in range(1000): 
			last = read_csv(output_url, header=None)[-1:]
			if last.values[0,0] != "#TRILEGAL normally terminated":
				print("...")
				sleep(10)
			elif last.values[0,0] == "#TRILEGAL normally terminated":
				break
		# find number of stars fainter than target within best resolution			
		df = read_csv(output_url, sep="\s+")[:-2]
		headers = np.array(list(df))
		# if we were able to use TRILEGAL v1.6 and get TESS mags, use them
		if "TESS" in headers:
			Tmags = df["TESS"].values
			# for i in range(df.shape[0]):
				# Tmags[i] = df["TESS"][i]
			N_fgbg = Tmags[Tmags >= Tmag].shape[0]
		# otherwise, use 2mass mags from TRILEGAL v1.5 and convert to T mags using the relations from section 2.2.1.1 of Stassun et al. 2018
		else:
			Tmags = np.zeros(df.shape[0])
			Jmags = df["J"].values
			Kmags = df["Ks"].values
			for i, (J, Ks) in enumerate(zip(Jmags, Kmags)):
				if (-0.1 <= J-Ks <= 0.70):
					Tmags[i] = Tmags[i] = J + 1.22163*(J-Ks)**3 - 1.74299*(J-Ks)**2 + 1.89115*(J-Ks) + 0.0563
				elif (0.7 < J-Ks <= 1.0):
					Tmags[i] = Tmags[i] = J - 269.372*(J-Ks)**3 + 668.453*(J-Ks)**2 - 545.64*(J-Ks) + 147.811
				elif (J-Ks < -0.1):
					Tmags[i] = J + 0.5
				elif (J-Ks > 1.0):
					Tmags[i] = J + 1.75
			N_fgbg = Tmags[Tmags >= Tmag].shape[0]
		f_fgbg = N_fgbg * 10/3600**2 * best_resolution**2
		return f_fgbg