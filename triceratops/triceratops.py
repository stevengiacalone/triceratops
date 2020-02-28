from astroquery.mast import Catalogs, Tesscut
from astropy.coordinates import SkyCoord
from astropy import constants
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
import numpy as np
from pandas import DataFrame, read_csv
from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from pkg_resources import resource_filename

from .priors import query_TRILEGAL, prior_TP, prior_EB, prior_bound_companion, prior_unbound_companion
from .likelihoods import simulate_TP_transit, simulate_EB_transit, likelihood_TTP, likelihood_TEB, likelihood_PTP, likelihood_STP, likelihood_PEB, likelihood_SEB, likelihood_DTP, likelihood_BTP, likelihood_DEB, likelihood_BEB, likelihood_uncharacterized_NTP, likelihood_uncharacterized_NEB
from .funcs import stellar_relations, contrast_curve, renorm_flux, renorm_flux_err, nearby_star_luminosity, color_Teff_relations, quad_coeffs

# EB data that is used by default
KEBC_FILE = resource_filename('triceratops','data/KeplerEBs.csv')
df = read_csv(KEBC_FILE, skiprows=7)
EBs = df[(df.Teff > 0) & (df.period <= 50) & (df.morph > 0) & (df.morph < 0.8) & (df.pdepth > 0.0)]

class target:
	def __init__(self, ID:int, sectors:np.ndarray, search_radius:int = 5):
		"""
		Queries TIC for sources near the target and obtains a cutout of the pixels enclosing the target.
		Args:
			ID (int): TIC ID of the target.
			sectors (numpy array): Sectors in which the target has been observed.
			search_radius (int): Number of pixels from the target star to search.
		"""

		self.ID = ID
		self.sectors = sectors
		self.search_radius = search_radius
		self.N_pix = 2*search_radius+2

		# query TIC for nearby stars
		pixel_size = 20.25*u.arcsec
		df = Catalogs.query_object("TIC"+str(ID), radius=search_radius*pixel_size, catalog="TIC")
		stars = df["ID", "Tmag", "ra", "dec", "mass", "rad", "Teff", "logg", "lum", "plx", "Vmag", "Kmag"].to_pandas()
		self.stars = stars

		TESS_images = []
		col0s, row0s = [], []
		pix_coords = []
		# for each sector, get FFI cutout and transform RA/Dec into TESS pixel coordinates
		for j,sector in enumerate(sectors):
			Tmag, ra, dec = stars["Tmag"].values, stars["ra"].values, stars["dec"].values
			cutout_coord = SkyCoord(ra[0], dec[0], unit="deg")
			cutout_hdu = Tesscut.get_cutouts(cutout_coord, size=self.N_pix, sector=sector)[0]
			cutout_table = cutout_hdu[1].data
			hdu = cutout_hdu[2].header
			wcs = WCS(hdu)
			# self.TESS_image = np.mean(cutout_table["FLUX"], axis=0)
			# self.col0, self.row0 = cutout_hdu[1].header["1CRV4P"], cutout_hdu[1].header["2CRV4P"]
			TESS_images.append(np.mean(cutout_table["FLUX"], axis=0))
			col0, row0 = cutout_hdu[1].header["1CRV4P"], cutout_hdu[1].header["2CRV4P"]
			col0s.append(col0)
			row0s.append(row0)

			pix_coord = np.zeros([len(ra),2])
			for i in range(len(ra)):
				RApix = np.asscalar(wcs.all_world2pix(ra[i],dec[i],0)[0])
				Decpix = np.asscalar(wcs.all_world2pix(ra[i],dec[i],0)[1])
				pix_coord[i,0] = col0+RApix
				pix_coord[i,1] = row0+Decpix    
			# self.pix_coords = pix_coord
			pix_coords.append(pix_coord)
		self.TESS_images = TESS_images
		self.col0s = col0s
		self.row0s = row0s
		self.pix_coords = pix_coords
		return

	def add_star(self, ID:int, Tmag:float, bound:bool):
		"""
		Adds an additional star (e.g., an unresolved star identified with follow up) to the table of stars.
		Args:
			ID (int): Arbitrary ID for the new star.
			Tmag (float): Estimated TESS magnitude of the new star.
			bound (bool): True if new star is physically bound to target star, False if new star is unbound.
		"""

		# if bound, use same parallax as target star
		if bound:
			plx = self.stars['plx'].values[0]
			new_star = DataFrame([[str(ID), Tmag, plx]], columns=["ID", "Tmag", "plx"])
		else:
			new_star = DataFrame([[str(ID), Tmag]], columns=["ID", "Tmag"])
		self.stars = self.stars.append(new_star)
		# for each set of pixel coordinates (corresponding to each sector), append a row for the new star with the same coordinates as the target star
		for i in range(len(self.pix_coords)):
			self.pix_coords[i] = np.append(self.pix_coords[i], self.pix_coords[i][0]).reshape(len(self.pix_coords[i])+1, 2)
		return

	def plot_field(self, sector:int = None, ap_pixels = None):
		"""
		Plots the field of stars and pixels around the target.
		Args:
			sector (int): Sector to plot.
			ap_pixels (numpy array): Aperture used to extract light curve. 
		"""
		
		if len(self.sectors) > 1:
			idx = np.argwhere(self.sectors == sector)[0,0]
		else:
			idx = 0

		corners = np.arange(-0.5,self.N_pix+0.5,1)
		centers = np.arange(0,self.N_pix,1)
		fig, ax = plt.subplots(1, 2, figsize=(15,6))
		plt.subplots_adjust(wspace=0.15)
		# aperture
		if ap_pixels is not None:
			for i in range(len(ap_pixels)):
				ax[0].fill_between([ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5], [ap_pixels[i][1]-0.5, ap_pixels[i][1]-0.5], [ap_pixels[i][1]+0.5, ap_pixels[i][1]+0.5], color="cyan", alpha=0.5)
		# pixel grid
		for i in corners:
			ax[0].plot(np.full_like(corners, self.col0s[idx]+i), self.row0s[idx]+corners, "k-", lw=0.5)
			ax[0].plot(self.col0s[idx]+corners, np.full_like(corners, self.row0s[idx]+i), "k-", lw=0.5)
		# search radius
		ax[0].plot(self.pix_coords[idx][0,0]+self.search_radius*np.cos(np.linspace(0,2*np.pi,100)), 
				   self.pix_coords[idx][0,1]+self.search_radius*np.sin(np.linspace(0,2*np.pi,100)), "k--", alpha=0.5)
		# stars
		sc = ax[0].scatter(self.pix_coords[idx][:,0], self.pix_coords[idx][:,1], c=self.stars["Tmag"], s=75, edgecolors="k", cmap=cm.viridis_r, 
						   vmin=floor(min(self.stars["Tmag"])), vmax=ceil(max(self.stars["Tmag"])))
		cb1 = fig.colorbar(sc, ax=ax[0], pad=0.02)
		cb1.ax.set_ylabel("T mag", rotation=270, fontsize=12, labelpad=18)
		for i in range(len(self.stars)):
			ax[0].annotate(str(i), (self.pix_coords[idx][i,0]-0.6, self.pix_coords[idx][i,1]-0.4), fontsize=12)
		ax[0].set_ylim([min(self.row0s[idx]+corners), max(self.row0s[idx]+corners)])
		ax[0].set_xlim([min(self.col0s[idx]+corners), max(self.col0s[idx]+corners)])
		ax[0].set_yticks(self.row0s[idx]+centers)
		ax[0].set_xticks(self.col0s[idx]+centers)
		ax[0].tick_params(width=0)
		ax[0].set_ylabel("pixel row number", fontsize=12)
		ax[0].set_xlabel("pixel column number", fontsize=12)
		# TESS FFI
		im = ax[1].imshow(self.TESS_images[idx], extent=[min(self.col0s[idx]+corners), max(self.col0s[idx]+corners), max(self.row0s[idx]+corners), min(self.row0s[idx]+corners)])
		cb2 = fig.colorbar(im, ax=ax[1], pad=0.02)
		cb2.ax.set_ylabel("flux [e$^-$ s$^{-1}$]", rotation=270, fontsize=12, labelpad=18)
		ax[1].set_ylim([min(self.row0s[idx]+corners), max(self.row0s[idx]+corners)])
		ax[1].set_xlim([min(self.col0s[idx]+corners), max(self.col0s[idx]+corners)])
		ax[1].set_yticks(self.row0s[idx]+centers)
		ax[1].set_xticks(self.col0s[idx]+centers)
		ax[1].tick_params(width=0)
		ax[1].set_ylabel("pixel row number", fontsize=12)
		ax[1].set_xlabel("pixel column number", fontsize=12)
		# aperture
		if ap_pixels is not None:
			for i in range(len(ap_pixels)):
				ax[1].fill_between([ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5], [ap_pixels[i][1]-0.5, ap_pixels[i][1]-0.5], [ap_pixels[i][1]+0.5, ap_pixels[i][1]+0.5], color="cyan", alpha=0.2)
		plt.show()
		return

	def calc_depths(self, tdepth:float, all_ap_pixels):
		"""
		Calculates the transit depth each source in the aperture would have if it were the source of the transit.
		Args:
			tdepth (float): Reported transit depth [%].
			all_ap_pixels (numpy array): Apertures used to extract light curve. 
		"""
		
		# find stars in all apertures
		ap_mask = np.zeros(len(self.stars), dtype=bool)
		for k in range(len(all_ap_pixels)):
			for i in range(len(self.stars)):
				for j in range(len(all_ap_pixels[k])):
					in_ap = np.logical_and((all_ap_pixels[k][j,0]-0.5 <= self.pix_coords[k][i,0] <= all_ap_pixels[k][j,0]+0.5), (all_ap_pixels[k][j,1]-0.5 <= self.pix_coords[k][i,1] <= all_ap_pixels[k][j,1]+0.5))
					if in_ap == True:
						ap_mask[i] = in_ap

		# calculate relative flux of each star using Tmags
		relative_flux = np.zeros(len(self.stars))
		for i in range(len(self.stars)):
			if ap_mask[i] == True:
				relative_flux[i] = 10**((np.min(self.stars.Tmag.values[ap_mask]) - self.stars.Tmag.values[i])/2.5)
			else:
				relative_flux[i] = 0
		# append flux ratio to stars dataframe
		flux_ratios = relative_flux/np.sum(relative_flux)
		self.stars["fluxratio"] = flux_ratios
		# calculate transit depth of each star given input transit depth
		tdepths = np.zeros(len(self.stars))
		for i in range(len(flux_ratios)):
			if flux_ratios[i] != 0:
				tdepths[i] = 1 - (flux_ratios[i] - tdepth) / flux_ratios[i]
		tdepths[tdepths > 1] = 0
		self.stars["tdepth"] = tdepths
		return

	def calc_probs(self, time, flux_0, flux_err_0, P_orb:float, contrast_curve_file:str = None):
		"""
		Calculates the relative probability of each scenario.
		Args:
			time (numpy array): Time of each data point [days from transit midpoint].
			flux_0 (numpy array): Normalized flux of each data point.
			flux_err_0 (numpy array): Normalized flux error of each data point.
			P_orb (float): Orbital period [days].
			contrast_curve_file (str): Path to contrast curve text file. File should contain column with separations (in arcsec)
										followed by column with Delta_mags.
		"""

		# construct a new dataframe that gives the values of lnL, best fit parameters, lnprior, and relative probability of each scenario considered 
		filtered_stars = self.stars[self.stars["tdepth"] > 0]
		N_scenarios = 2*len(filtered_stars) + 8
		targets = np.zeros(N_scenarios, dtype=np.dtype("i8"))
		case = np.zeros(N_scenarios, dtype=np.dtype('U1'))
		star_num = np.zeros(N_scenarios, dtype=np.dtype("i8"))
		scenarios = np.zeros(N_scenarios, dtype=np.dtype('U3'))
		best_i = np.zeros(N_scenarios)
		best_rp = np.zeros(N_scenarios)
		best_fluxratio = np.zeros(N_scenarios)
		best_Delta_mags = np.zeros(N_scenarios)
		best_lum = np.zeros(N_scenarios)
		best_Teff = np.zeros(N_scenarios)
		best_ms = np.zeros(N_scenarios)
		best_rs = np.zeros(N_scenarios)
		lnL = np.zeros(N_scenarios)
		lnprior = np.zeros(N_scenarios)

		for i, ID in enumerate(filtered_stars["ID"].values):
			# subtract flux from other stars in the aperture
			flux = renorm_flux(flux_0, filtered_stars["fluxratio"].values[i], True)
			flux_err = renorm_flux_err(flux_err_0, filtered_stars["fluxratio"].values[i], True)

			# target star
			if i == 0:
				# start TRILEGAL query
				output_url = query_TRILEGAL(filtered_stars["ra"].values[i], filtered_stars["dec"].values[i])

				# check to see if there are any missing stellar parameters and, if there are, estimate them given at least one
				if np.isnan(filtered_stars["mass"].values[i]) == True or np.isnan(filtered_stars["rad"].values[i]) == True or np.isnan(filtered_stars["Teff"].values[i]) == True or np.isnan(filtered_stars["logg"].values[i]) == True or np.isnan(filtered_stars["lum"].values[i]) == True:
					done = 0
					if np.isnan(filtered_stars["mass"].values[i]) == True and np.isnan(filtered_stars["rad"].values[i]) == True and np.isnan(filtered_stars["Teff"].values[i]) == True and np.isnan(filtered_stars["logg"].values[i]) == True and np.isnan(filtered_stars["lum"].values[i]) == True:
						print("Insufficient information to validate " + str(ID) + ". Please provide an estimate for stellar mass, radius, Teff, logg, or luminosity.")
						break
					elif not np.isnan(filtered_stars["Teff"].values[i]) and done == 0:
						print("Estimating stellar properties for " + str(ID) + " based on Teff.")
						filtered_stars.at[i, "mass"] = stellar_relations(Teff=filtered_stars["Teff"].values[i])[0]
						filtered_stars.at[i, "rad"] = stellar_relations(Teff=filtered_stars["Teff"].values[i])[1]
						filtered_stars.at[i, "logg"] = stellar_relations(Teff=filtered_stars["Teff"].values[i])[3]
						filtered_stars.at[i, "lum"] = stellar_relations(Teff=filtered_stars["Teff"].values[i])[4]
						done = 1
					elif not np.isnan(filtered_stars["lum"].values[i]) and (filtered_stars["lum"].values[i] > 0.0) and done == 0:
						print("Estimating stellar properties for " + str(ID) + " based on luminosity.")
						filtered_stars.at[i, "mass"] = stellar_relations(lum=filtered_stars["lum"].values[i])[0]
						filtered_stars.at[i, "rad"] = stellar_relations(lum=filtered_stars["lum"].values[i])[1]
						filtered_stars.at[i, "Teff"] = stellar_relations(lum=filtered_stars["lum"].values[i])[2]
						filtered_stars.at[i, "logg"] = stellar_relations(lum=filtered_stars["lum"].values[i])[3]
						done = 1
					elif not np.isnan(filtered_stars["rad"].values[i]) and done == 0:
						print("Estimating stellar properties for " + str(ID) + " based on radius.")
						filtered_stars.at[i, "mass"] = stellar_relations(Rad=filtered_stars["rad"].values[i])[0]
						filtered_stars.at[i, "Teff"] = stellar_relations(Rad=filtered_stars["rad"].values[i])[2]
						filtered_stars.at[i, "logg"] = stellar_relations(Rad=filtered_stars["rad"].values[i])[3]
						filtered_stars.at[i, "lum"] = stellar_relations(Rad=filtered_stars["rad"].values[i])[4]
						done = 1
					elif not np.isnan(filtered_stars["mass"].values[i]) and done == 0:
						print("Estimating stellar properties for " + str(ID) + " based on mass.")
						filtered_stars.at[i, "rad"] = stellar_relations(Mass=filtered_stars["mass"].values[i])[1]
						filtered_stars.at[i, "Teff"] = stellar_relations(Mass=filtered_stars["mass"].values[i])[2]
						filtered_stars.at[i, "logg"] = stellar_relations(Mass=filtered_stars["mass"].values[i])[3]
						filtered_stars.at[i, "lum"] = stellar_relations(Mass=filtered_stars["mass"].values[i])[4]
						done = 1

				print("Calculating TTP and TEB scenario probabilities for " + str(ID) + ".")

				j=0
				targets[i+j] = ID
				case[i+j] = "A"
				star_num[i+j] = 1
				scenarios[i+j] = "TTP"
				lnL[i+j], best_i[i+j], best_rp[i+j] = likelihood_TTP(time, flux, flux_err, P_orb, filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["logg"].values[i])
				best_fluxratio[i+j] = 0.0
				best_Delta_mags[i+j] = -np.inf
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], best_lum[i+j] = filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["lum"].values[i]


				j=1
				targets[i+j] = ID
				case[i+j] = "A"
				star_num[i+j] = 1
				scenarios[i+j] = "TEB"
				lnL[i+j], best_i[i+j], best_EB_fluxratio = likelihood_TEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+1] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio))
				x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio))
				best_fluxratio[i+j] = 0.0
				best_Delta_mags[i+j] = -np.inf

				print("Calculating PTP, PEB, STP, and SEB scenario probabilities for " + str(ID) + ".")

				j=2
				targets[i+j] = ID
				case[i+j] = "B"
				star_num[i+j] = 1
				scenarios[i+j] = "PTP"
				lnL[i+j], best_i[i+j], best_rp[i+j], best_fluxratio[i+j] = likelihood_PTP(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_fluxratio[i+j]))


				j=3
				targets[i+j] = ID
				case[i+j] = "B"
				star_num[i+j] = 1
				scenarios[i+j] = "PEB"
				lnL[i+j], best_i[i+j], best_EB_fluxratio, best_fluxratio[i+j] = likelihood_PEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio)*(1-best_fluxratio[i+j]))
				x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio)*(1-best_fluxratio[i+j]))


				j=4
				targets[i+j] = ID
				case[i+j] = "B"
				star_num[i+j] = 2
				scenarios[i+j] = "STP"
				lnL[i+j], best_i[i+j], best_rp[i+j], best_fluxratio[i+j] = likelihood_STP(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_fluxratio[i+j]))
				primary_ms_STP, x, x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_fluxratio[i+j]))


				j=5
				targets[i+j] = ID
				case[i+j] = "B"
				star_num[i+j] = 2
				scenarios[i+j] = "SEB"
				lnL[i+j], best_i[i+j], best_EB_fluxratio, best_fluxratio[i+j] = likelihood_SEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio)*(best_fluxratio[i+j]))
				x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio)*(best_fluxratio[i+j]))
				primary_ms_SEB, x, x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_fluxratio[i+j]))

				print("Calculating DTP, DEB, BTP, and BEB scenario probabilities for " + str(ID) + ".")

				j=6
				targets[i+j] = ID
				case[i+j] = "C"
				star_num[i+j] = 1
				scenarios[i+j] = "DTP"
				lnL[i+j], best_i[i+j], best_rp[i+j], best_fluxratio[i+j] = likelihood_DTP(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_fluxratio[i+j]))
				

				j=7
				targets[i+j] = ID
				case[i+j] = "C"
				star_num[i+j] = 1
				scenarios[i+j] = "DEB"
				lnL[i+j], best_i[i+j], best_EB_fluxratio, best_fluxratio[i+j] = likelihood_DEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i])
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio)*(1-best_fluxratio[i+j]))
				x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio)*(1-best_fluxratio[i+j]))
				

				j=8
				targets[i+j] = ID
				case[i+j] = "C"
				star_num[i+j] = 2
				scenarios[i+j] = "BTP"
				lnL[i+j], best_i[i+j], best_rp[i+j], best_fluxratio[i+j], best_Teff[i+j] = likelihood_BTP(time, flux, flux_err, P_orb)
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				best_ms[i+j], best_rs[i+j], x, x, best_lum[i+j] = stellar_relations(Teff=best_Teff[i+j])
				primary_ms_BTP, x, x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_fluxratio[i+j]))


				j=9
				targets[i+j] = ID
				case[i+j] = "C"
				star_num[i+j] = 2
				scenarios[i+j] = "BEB"
				lnL[i+j], best_i[i+j], best_EB_fluxratio, best_fluxratio[i+j], fgbg_Teff_guess = likelihood_BEB(time, flux, flux_err, P_orb)
				best_Delta_mags[i+j] = np.abs(2.5*np.log10( best_fluxratio[i+j] / (1 - best_fluxratio[i+j]) ))
				fgbg_lum_guess = stellar_relations(Teff=fgbg_Teff_guess)[-1]
				best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=fgbg_lum_guess*(1-best_EB_fluxratio))
				x, best_rp[i+j], x, x, x = stellar_relations(lum=fgbg_lum_guess*(best_EB_fluxratio))
				primary_ms_BEB, x, x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_fluxratio[i+j]))
					

				# compute priors
				j=0
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], 2.2)
				companion_prior = (1-prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], 2.2)) * (1-unbound_companion_prior)
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))
				else:
					lnprior[i+j] = -np.inf
				j=1
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], 2.2)
				companion_prior = (1-prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], 2.2)) * (1-unbound_companion_prior)
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))
				else:
					lnprior[i+j] = -np.inf
				j=2
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], 2.2)
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				companion_prior = prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], best_res) * (1-unbound_companion_prior)
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))
				else:
					lnprior[i+j] = -np.inf
				j=3
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], 2.2)
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				companion_prior = prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], best_res) * (1-unbound_companion_prior)
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))
				else:
					lnprior[i+j] = -np.inf
				j=4
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], 2.2)
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				companion_prior = prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], best_res) * (1-unbound_companion_prior)
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))
				else:
					lnprior[i+j] = -np.inf
				j=5
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], 2.2)
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				companion_prior = prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], best_res) * (1-unbound_companion_prior)
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))
				else:
					lnprior[i+j] = -np.inf
				j=6
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], best_res)
				companion_prior = (1-prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], 2.2)) * unbound_companion_prior
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))
				else:
					lnprior[i+j] = -np.inf
				j=7
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], best_res)
				companion_prior = (1-prior_bound_companion(best_ms[i+j], filtered_stars["plx"].values[i], 2.2)) * unbound_companion_prior
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))
				else:
					lnprior[i+j] = -np.inf
				j=8
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], best_res)
				companion_prior = (1-prior_bound_companion(primary_ms_BTP, filtered_stars["plx"].values[i], 2.2)) * unbound_companion_prior
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))
				else:
					lnprior[i+j] = -np.inf
				j=9
				best_res = contrast_curve(contrast_curve_file, best_Delta_mags[i+j])
				unbound_companion_prior = prior_unbound_companion(output_url, filtered_stars["Tmag"].values[i], best_res)
				companion_prior = (1-prior_bound_companion(primary_ms_BEB, filtered_stars["plx"].values[i], 2.2)) * unbound_companion_prior
				if companion_prior > 0.0:
					lnprior[i+j] = np.log(companion_prior * prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))
				else:
					lnprior[i+j] = -np.inf

			# nearby stars
			else:
				if (np.isnan(filtered_stars["rad"].values[i]) == False and np.isnan(filtered_stars["Teff"].values[i]) == False) and np.isnan(filtered_stars["mass"].values[i]) == True and np.isnan(filtered_stars["logg"].values[i]) == True and np.isnan(filtered_stars["lum"].values[i]) == True:
					print("Skipping " + str(ID) + ". Evolved star.")
					j=10 + (i-2)
					lnprior[i+j] = -np.inf
					lnL[i+j] = np.inf
					j=11 + (i-2)
					lnprior[i+j] = -np.inf
					lnL[i+j] = np.inf
					continue

				# check to see if there are stellar parameters we can use - if any are missing, assume MS and estimate properties based on TESS magnitude and parallax OR based on V and Ks
				if np.isnan(filtered_stars["mass"].values[i]) or np.isnan(filtered_stars["rad"].values[i]) or np.isnan(filtered_stars["Teff"].values[i]) or np.isnan(filtered_stars["logg"].values[i]) or np.isnan(filtered_stars["lum"].values[i]):
					
					# first, try based on TESS magnitude and parallax
					if ~np.isnan(filtered_stars["Tmag"].values[i]) and ~np.isnan(filtered_stars["plx"].values[i]) and ~np.isnan(filtered_stars["Tmag"].values[0]) and ~np.isnan(filtered_stars["plx"].values[0]) and ~np.isnan(filtered_stars["lum"].values[0]):
						print(str(ID) + " missing info. Estimaing properties based on TESS mag and parallax.")
						L_est = nearby_star_luminosity(filtered_stars["Tmag"].values[0], filtered_stars["Tmag"].values[i], filtered_stars["plx"].values[0], filtered_stars["plx"].values[i], filtered_stars["lum"].values[0])
						filtered_stars["lum"].values[i] = L_est
						filtered_stars["mass"].values[i] = stellar_relations(lum=L_est)[0]
						filtered_stars["rad"].values[i] = stellar_relations(lum=L_est)[1]
						filtered_stars["Teff"].values[i] = stellar_relations(lum=L_est)[2]
						filtered_stars["logg"].values[i] = stellar_relations(lum=L_est)[3]

						print("Calculating NTP and NEB scenario probabilities for " + str(ID) + ".")

						j=10 + (i-2)
						targets[i+j] = ID
						case[i+j] = "A"
						star_num[i+j] = 1
						scenarios[i+j] = "NTP"
						lnL[i+j], best_i[i+j], best_rp[i+j] = likelihood_TTP(time, flux, flux_err, P_orb, filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["logg"].values[i])
						best_fluxratio[i+j] = 0.0
						best_Delta_mags[i+j] = -np.inf
						best_ms[i+j], best_rs[i+j], best_Teff[i+j], best_lum[i+j] = filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["lum"].values[i]
						lnprior[i+j] = np.log(prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))

						j=11 + (i-2)
						targets[i+j] = ID
						case[i+j] = "A"
						star_num[i+j] = 1
						scenarios[i+j] = "NEB"
						lnL[i+j], best_i[i+j], best_EB_fluxratio = likelihood_TEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i],)
						best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio))
						x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio))
						best_fluxratio[i+j] = 0.0
						best_Delta_mags[i+j] = -np.inf
						lnprior[i+j] = np.log(prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))

					# if above doesn't work, use V and Ks mags
					elif ~np.isnan(filtered_stars["Vmag"].values[i]) and ~np.isnan(filtered_stars["Kmag"].values[i]):
						print(str(ID) + " missing info. Estimaing properties based on V-Ks color.")
						Teff_est = color_Teff_relations(filtered_stars["Vmag"].values[i], filtered_stars["Kmag"].values[i])
						filtered_stars["Teff"].values[i] = Teff_est
						filtered_stars["mass"].values[i] = stellar_relations(Teff=Teff_est)[0]
						filtered_stars["rad"].values[i] = stellar_relations(Teff=Teff_est)[1]
						filtered_stars["logg"].values[i] = stellar_relations(Teff=Teff_est)[3]
						filtered_stars["lum"].values[i] = stellar_relations(Teff=Teff_est)[4]

						print("Calculating NTP and NEB scenario probabilities for " + str(ID) + ".")

						j=10 + (i-2)
						targets[i+j] = ID
						case[i+j] = "A"
						star_num[i+j] = 1
						scenarios[i+j] = "NTP"
						lnL[i+j], best_i[i+j], best_rp[i+j] = likelihood_TTP(time, flux, flux_err, P_orb, filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["logg"].values[i])
						best_fluxratio[i+j] = 0.0
						best_Delta_mags[i+j] = -np.inf
						best_ms[i+j], best_rs[i+j], best_Teff[i+j], best_lum[i+j] = filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["lum"].values[i]
						lnprior[i+j] = np.log(prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))

						j=11 + (i-2)
						targets[i+j] = ID
						case[i+j] = "A"
						star_num[i+j] = 1
						scenarios[i+j] = "NEB"
						lnL[i+j], best_i[i+j], best_EB_fluxratio = likelihood_TEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i],)
						best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio))
						x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio))
						best_fluxratio[i+j] = 0.0
						best_Delta_mags[i+j] = -np.inf
						lnprior[i+j] = np.log(prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))

					# last resort, assume MS and try a bunch of spectral types
					else:
						print(str(ID) + " missing info. Trying a few different spectral types.")

						print("Calculating NTP and NEB scenario probabilities for " + str(ID) + ".")

						j=10 + (i-2)
						targets[i+j] = ID
						case[i+j] = "A"
						star_num[i+j] = 1
						scenarios[i+j] = "NTP"
						lnL[i+j], best_i[i+j], best_rp[i+j], best_Teff[i+j] = likelihood_uncharacterized_NTP(time, flux, flux_err, P_orb)
						best_ms[i+j], best_rs[i+j], x, x, best_lum[i+j] = stellar_relations(Teff=best_Teff[i+j])
						best_fluxratio[i+j] = 0.0
						best_Delta_mags[i+j] = -np.inf
						lnprior[i+j] = np.log(prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))

						j=11 + (i-2)
						targets[i+j] = ID
						case[i+j] = "A"
						star_num[i+j] = 1
						scenarios[i+j] = "NEB"
						lnL[i+j], best_i[i+j], best_EB_fluxratio, fgbg_Teff_guess = likelihood_uncharacterized_NEB(time, flux, flux_err, P_orb)
						fgbg_lum_guess = stellar_relations(Teff=fgbg_Teff_guess)[-1]
						best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=fgbg_lum_guess*(1-best_EB_fluxratio))
						x, best_rp[i+j], x, x, x = stellar_relations(lum=fgbg_lum_guess*(best_EB_fluxratio))
						best_fluxratio[i+j] = 0.0
						best_Delta_mags[i+j] = -np.inf
						lnprior[i+j] = np.log(prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))


				else:
					print("Calculating NTP and NEB scenario probabilities for " + str(ID) + ".")

					j=10 + (i-2)
					targets[i+j] = ID
					case[i+j] = "A"
					star_num[i+j] = 1
					scenarios[i+j] = "NTP"
					lnL[i+j], best_i[i+j], best_rp[i+j] = likelihood_TTP(time, flux, flux_err, P_orb, filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["logg"].values[i])
					best_fluxratio[i+j] = 0.0
					best_Delta_mags[i+j] = -np.inf
					best_ms[i+j], best_rs[i+j], best_Teff[i+j], best_lum[i+j] = filtered_stars["mass"].values[i], filtered_stars["rad"].values[i], filtered_stars["Teff"].values[i], filtered_stars["lum"].values[i]
					lnprior[i+j] = np.log(prior_TP(P_orb, (best_rp[i+j]*(constants.R_earth.cgs.value/constants.R_sun.cgs.value)/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j]))

					j=11 + (i-2)
					targets[i+j] = ID
					case[i+j] = "A"
					star_num[i+j] = 1
					scenarios[i+j] = "NEB"
					lnL[i+j], best_i[i+j], best_EB_fluxratio = likelihood_TEB(time, flux, flux_err, P_orb, filtered_stars["lum"].values[i],)
					best_ms[i+j], best_rs[i+j], best_Teff[i+j], x, best_lum[i+j] = stellar_relations(lum=filtered_stars["lum"].values[i]*(1-best_EB_fluxratio))
					x, best_rp[i+j], x, x, x = stellar_relations(lum=filtered_stars["lum"].values[i]*(best_EB_fluxratio))
					best_fluxratio[i+j] = 0.0
					best_Delta_mags[i+j] = -np.inf
					lnprior[i+j] = np.log(prior_EB(P_orb, (best_rp[i+j]/best_rs[i+j])**2, best_ms[i+j], best_rs[i+j], EBs["period"], EBs["pdepth"]))

		# normalize likelihoods
		lnL *= -1
		lnL += np.min(np.abs(lnL))

		# calculate the relative probability of each scenario
		relative_probs = np.zeros(N_scenarios)
		for i in range(N_scenarios):
			relative_probs[i] = (np.exp(lnL[i]+lnprior[i])) / np.sum(np.exp(lnL+lnprior))

		# now save all of the arrays as a dataframe
		prob_df = DataFrame({"ID": targets, "scenario": scenarios,"ms": best_ms, "rs": best_rs, "Teff": best_Teff, "lum": best_lum, "fluxratio": best_fluxratio, "Delta_mag": best_Delta_mags, "inc": best_i, "rp": best_rp, "lnL": lnL, "lnprior": lnprior, "prob": relative_probs})
		self.probs = prob_df
		self.star_num = star_num

		# calculate the FPP
		self.FPP = 1 - (prob_df.prob[0] + prob_df.prob[2] + prob_df.prob[6])

		return

	def plot_fits(self, time, flux_0, flux_err_0, P_orb:float):
		"""
		Plots light curve for best fit instance of each scenario.
		Args:
			time (numpy array): Time of each data point [days from transit midpoint].
			flux_0 (numpy array): Normalized flux of each data point.
			flux_err_0 (numpy array): Normalized flux error of each data point.
			P_orb (float): Orbital period [days].
		"""

		scenario_idx = self.probs[self.probs["ID"] != 0].index.values
		df = self.probs[self.probs["ID"] != 0]
		star_num = self.star_num[self.probs["ID"] != 0]
		f, ax = plt.subplots(len(df)//2, 2, figsize=(12,len(df)//2*4), sharex=True)
		for i in range(len(df)//2):
			for j in range(2):
				if i == 0:
					k = j
				else:
					k = 2*i+j
				# subtract flux from other stars in the aperture
				idx = np.argwhere(self.stars["ID"].values == str(df["ID"].values[k]))[0,0]
				flux = renorm_flux(flux_0, self.stars["fluxratio"].values[idx], True)
				flux_err = renorm_flux_err(flux_err_0, self.stars["fluxratio"].values[idx], True)
				# all TPs
				if j == 0:
					# if df["star"].values[k] == 1:
					if star_num[k] == 1:
						comp = False
					else:
						comp = True
					a = ((constants.G.cgs.value*df["ms"].values[k]*constants.M_sun.cgs.value)/(4*np.pi**2)*(P_orb*86400)**2)**(1/3)
					logg = np.log10(constants.G.cgs.value*df["ms"].values[k]*constants.M_sun.cgs.value/(df["rs"].values[k]*constants.R_sun.cgs.value)**2)
					u1, u2 = quad_coeffs(df["Teff"].values[k], logg, 0.0)
					best_model = simulate_TP_transit((df["inc"].values[k], df["rp"].values[k]), time, P_orb, a, df["rs"].values[k], u1, u2)
					scenario_flux = renorm_flux(flux, df["fluxratio"].values[k], comp)
					scenario_flux_err = renorm_flux_err(flux_err, df["fluxratio"].values[k], comp)
				# all EBs
				else:
					# if df["star"].values[k] == 1:
					if star_num[k] == 1:
						comp = False
					else:
						comp = True
					lum_EB = stellar_relations(Rad=df["rp"].values[k])[-1]
					lum_primary = stellar_relations(Rad=df["rs"].values[k])[-1]
					EB_fluxratio = lum_EB/(lum_EB+lum_primary)
					best_model = simulate_EB_transit((df["inc"].values[k], EB_fluxratio), time, P_orb, lum_EB+lum_primary)
					scenario_flux = renorm_flux(flux, df["fluxratio"].values[k], comp)
					scenario_flux_err = renorm_flux_err(flux_err, df["fluxratio"].values[k], comp)
			
				y_formatter = ticker.ScalarFormatter(useOffset=False)
				ax[i,j].yaxis.set_major_formatter(y_formatter)
				ax[i,j].errorbar(time, scenario_flux, scenario_flux_err, fmt=".", color="blue", alpha=0.2, zorder=0)
				ax[i,j].plot(time, best_model, "k-", lw=5, zorder=2)
				ax[i,j].set_ylabel("normalized flux", fontsize=12)
				ax[i,j].annotate(str(scenario_idx[k]), xy=(0.05,0.05), xycoords="axes fraction", fontsize=12)
		ax[len(df)//2-1, 0].set_xlabel("days from transit center", fontsize=12)
		ax[len(df)//2-1, 1].set_xlabel("days from transit center", fontsize=12)
		plt.tight_layout()
		plt.show()
		return