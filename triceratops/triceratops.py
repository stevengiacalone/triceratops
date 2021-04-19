from astroquery.mast import Catalogs, Tesscut
from astropy.coordinates import SkyCoord
from astropy import constants
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
import numpy as np
from scipy.integrate import dblquad
from pandas import DataFrame, read_csv
from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from mpl_toolkits.axes_grid1.anchored_artists import (
    AnchoredDirectionArrows
    )

from .likelihoods import (simulate_TP_transit,
                         simulate_EB_transit)
from .funcs import (Gauss2D,
                   query_TRILEGAL,
                   renorm_flux,
                   stellar_relations)
from .marginal_likelihoods import *

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi
ln2pi = np.log(2*pi)


class target:
    def __init__(self, ID: int, sectors: np.ndarray,
                 search_radius: int = 10):
        """
        Queries TIC for sources near the target and obtains a cutout
        of the pixels enclosing the target.
        Args:
            ID (int): TIC ID of the target.
            sectors (numpy array): Sectors in which the target
                                   has been observed.
            search_radius (int): Number of pixels from the target
                                 star to search.
        """
        self.ID = ID
        self.sectors = sectors
        self.search_radius = search_radius
        self.N_pix = 2*search_radius+2
        # query TIC for nearby stars
        pixel_size = 20.25*u.arcsec
        df = Catalogs.query_object(
            "TIC"+str(ID),
            radius=search_radius*pixel_size,
            catalog="TIC"
            )
        new_df = df[
            "ID", "Tmag", "Jmag", "Hmag", "Kmag", "ra", "dec", "mass", "rad", "Teff", "plx"
            ]
        stars = new_df.to_pandas()

        # start TRILEGAL query
        output_url = query_TRILEGAL(
            stars["ra"].values[0],
            stars["dec"].values[0],
            verbose=0
            )
        self.trilegal_url = output_url

        TESS_images = []
        col0s, row0s = [], []
        pix_coords = []
        # for each sector, get FFI cutout and transform RA/Dec into
        # TESS pixel coordinates
        for j, sector in enumerate(sectors):
            Tmag = stars["Tmag"].values
            ra = stars["ra"].values
            dec = stars["dec"].values
            cutout_coord = SkyCoord(ra[0], dec[0], unit="deg")
            cutout_hdu = Tesscut.get_cutouts(
                cutout_coord,
                size=self.N_pix,
                sector=sector
                )[0]
            cutout_table = cutout_hdu[1].data
            hdu = cutout_hdu[2].header
            wcs = WCS(hdu)
            TESS_images.append(np.mean(cutout_table["FLUX"], axis=0))
            col0 = cutout_hdu[1].header["1CRV4P"]
            row0 = cutout_hdu[1].header["2CRV4P"]
            col0s.append(col0)
            row0s.append(row0)

            pix_coord = np.zeros([len(ra), 2])
            for i in range(len(ra)):
                RApix = np.asscalar(
                    wcs.all_world2pix(ra[i], dec[i], 0)[0]
                    )
                Decpix = np.asscalar(
                    wcs.all_world2pix(ra[i], dec[i], 0)[1]
                    )
                pix_coord[i, 0] = col0+RApix
                pix_coord[i, 1] = row0+Decpix
            pix_coords.append(pix_coord)

        # for each star, get the separation and position angle
        # from the targets star
        sep = [0]
        pa = [0]
        c_target = SkyCoord(
            stars["ra"].values[0],
            stars["dec"].values[0],
            unit="deg"
            )
        for i in range(1,len(stars)):
            c_star = SkyCoord(
                stars["ra"].values[i],
                stars["dec"].values[i],
                unit="deg"
                )
            sep.append(
                np.round(
                    c_target.separation(c_star).to(u.arcsec).value,
                    3
                    )
                )
            pa.append(
                np.round(
                    c_target.position_angle(c_star).to(u.deg).value,
                    3
                    )
                )
        stars["sep (arcsec)"] = sep
        stars["PA (E of N)"] = pa

        self.stars = stars
        self.TESS_images = TESS_images
        self.col0s = col0s
        self.row0s = row0s
        self.pix_coords = pix_coords
        return

    def add_star(self, ID: int, Tmag: float, bound: bool):
        """
        Adds an additional star (e.g., an unresolved star identified
        with follow up) to the table of stars.
        Args:
            ID (int): Arbitrary ID for the new star.
            Tmag (float): Estimated TESS magnitude of the new star.
            bound (bool): True if new star is physically bound to
                          target star, False if new star is unbound.
        """
        # if bound, use same parallax as target star
        if bound:
            plx = self.stars['plx'].values[0]
            new_star = DataFrame(
                [[str(ID), Tmag, plx]], columns=["ID", "Tmag", "plx"]
                )
        else:
            new_star = DataFrame(
                [[str(ID), Tmag]], columns=["ID", "Tmag"]
                )
        self.stars = self.stars.append(new_star)
        # for each set of pixel coordinates (corresponding to
        # each sector), append a row for the new star with the same
        # coordinates as the target star
        for i in range(len(self.pix_coords)):
            self.pix_coords[i] = np.append(
                self.pix_coords[i],
                self.pix_coords[i][0]
                ).reshape(
                self.pix_coords[i]+1, 2
                )
        return

    def plot_field(self, sector: int = None, ap_pixels=None,
                   ap_color: str = "red", save: bool = False,
                   fname: str = None):
        """
        Plots the field of stars and pixels around the target.
        Args:
            sector (int): Sector to plot.
            ap_pixels (numpy array): Aperture used to
                                     extract light curve.
            ap_color (str): Color of aperture outline.
            save (bool): Whether or not to save plot as pdf.
            fname (str): File name of pdf.
        """
        if len(self.sectors) > 1:
            idx = np.argwhere(self.sectors == sector)[0, 0]
        else:
            idx = 0

        corners = np.arange(-0.5, self.N_pix+0.5, 1)
        centers = np.arange(0, self.N_pix, 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        plt.subplots_adjust(right=0.9)
        # aperture
        if ap_pixels is not None:
            for i in range(len(ap_pixels)):
                ax[0].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]-0.5],
                    color=ap_color, zorder=1
                    )
                ax[0].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]+0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=1
                    )
                ax[0].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]-0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=1
                    )
                ax[0].plot(
                    [ap_pixels[i][0]+0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=1
                    )
        # pixel grid
        for i in corners:
            ax[0].plot(
                np.full_like(corners, self.col0s[idx]+i),
                self.row0s[idx]+corners, "k-", lw=0.5,
                zorder=0
                )
            ax[0].plot(
                self.col0s[idx]+corners,
                np.full_like(corners, self.row0s[idx]+i), "k-", lw=0.5,
                zorder=0
                )
        # search radius
        ax[0].plot(
            (
                self.pix_coords[idx][0, 0]
                + self.search_radius
                * np.cos(np.linspace(0, 2*pi, 100))
            ),
            (
                self.pix_coords[idx][0, 1]
                + self.search_radius
                * np.sin(np.linspace(0, 2*pi, 100))
            ),
            "k--", alpha=0.5, zorder=0)
        # N and E arrows
        v1 = np.array([0, 1])
        v2 = self.pix_coords[idx][1] - self.pix_coords[idx][0]
        sign = np.sign(v2[0])
        angle1 = sign * (
            np.arccos(np.dot(v1, v2)
            / np.sqrt((np.dot(v1, v1)) * np.dot(v2, v2))) * 180/np.pi
            )
        angle2 = self.stars["PA (E of N)"].values[1]
        rot = angle1-angle2
        roatated_arrow = AnchoredDirectionArrows(
                            ax[0].transAxes,
                            "E", "N",
                            loc="upper left",
                            color="k",
                            angle=-rot,
                            length=0.1,
                            fontsize=0.05,
                            back_length=0,
                            head_length=5,
                            head_width=5,
                            tail_width=1
                            )
        ax[0].add_artist(roatated_arrow)
        # stars
        sc = ax[0].scatter(
            self.pix_coords[idx][1:, 0],
            self.pix_coords[idx][1:, 1],
            c=self.stars["Tmag"].values[1:], s=75,
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=floor(min(self.stars["Tmag"])),
            vmax=ceil(max(self.stars["Tmag"])),
            zorder=2,
            rasterized=True
            )
        ax[0].scatter(
            [self.pix_coords[idx][0, 0]],
            [self.pix_coords[idx][0, 1]],
            c=[self.stars["Tmag"].values[0]], s=250,
            marker="*",
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=floor(min(self.stars["Tmag"])),
            vmax=ceil(max(self.stars["Tmag"])),
            zorder=2
            )
        cb1 = fig.colorbar(sc, ax=ax[0], pad=0.02)
        cb1.ax.set_ylabel(
            "TESS mag", rotation=270, fontsize=12, labelpad=18
            )
        ax[0].set_ylim([
            min(self.row0s[idx]+corners),
            max(self.row0s[idx]+corners)
            ])
        ax[0].set_xlim([
            min(self.col0s[idx]+corners),
            max(self.col0s[idx]+corners)
            ])
        ax[0].set_yticks(self.row0s[idx]+centers)
        ax[0].set_xticks(self.col0s[idx]+centers)
        ax[0].tick_params(width=0)
        ax[0].tick_params(axis='x', labelrotation=90)
        ax[0].set_ylabel("pixel row number", fontsize=12)
        ax[0].set_xlabel("pixel column number", fontsize=12)
        # TESS FFI
        im = ax[1].imshow(
            self.TESS_images[idx],
            extent=[
                min(self.col0s[idx]+corners),
                max(self.col0s[idx]+corners),
                max(self.row0s[idx]+corners),
                min(self.row0s[idx]+corners)
            ])
        cb2 = fig.colorbar(im, ax=ax[1], pad=0.02)
        cb2.ax.set_ylabel(
            "flux [e$^-$ s$^{-1}$]",
            rotation=270, fontsize=12, labelpad=18)
        ax[1].set_ylim([
            min(self.row0s[idx]+corners),
            max(self.row0s[idx]+corners)
            ])
        ax[1].set_xlim([
            min(self.col0s[idx]+corners),
            max(self.col0s[idx]+corners)
            ])
        ax[1].set_yticks(self.row0s[idx]+centers)
        ax[1].set_xticks(self.col0s[idx]+centers)
        ax[1].tick_params(width=0)
        ax[1].tick_params(axis='x', labelrotation=90)
        ax[1].set_ylabel("pixel row number", fontsize=12)
        ax[1].set_xlabel("pixel column number", fontsize=12)
        # aperture
        if ap_pixels is not None:
            for i in range(len(ap_pixels)):
                ax[1].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]-0.5],
                    color=ap_color, zorder=2
                    )
                ax[1].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]+0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=2
                    )
                ax[1].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]-0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=2
                    )
                ax[1].plot(
                    [ap_pixels[i][0]+0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=2
                    )
        if save is False:
            plt.tight_layout()
            plt.show()
        elif (save is True) & (fname is None):
            plt.tight_layout()
            target_star = self.stars.ID.values[0]
            plt.savefig("TIC"+str(target_star)+"_sector"+str(sector)+".pdf")
        else:
            plt.tight_layout()
            plt.savefig(fname+".pdf")
        return

    def calc_depths(self, tdepth: float, all_ap_pixels):
        """
        Calculates the transit depth each source near the target would
        have if it were the source of the transit.
        This is done by modeling the PSF of each source as a circular
        Gaussian with a standard deviation of 0.75 pixels.
        Args:
            tdepth (float): Reported transit depth [ppm].
            all_ap_pixels (list of numpy arrays): Apertures used to
                                                  extract light curve.
        """
        # for each aperture, calculate contribution due to each star
        rel_flux_per_aperture = np.zeros([
            len(all_ap_pixels),
            len(self.stars)
            ])
        flux_ratio_per_aperture = np.zeros([
            len(all_ap_pixels),
            len(self.stars)
            ])
        for k in range(len(all_ap_pixels)):
            for i in range(len(self.stars)):
                # location of star in pixel space for aperture k
                mu_x = self.pix_coords[k][i, 0]
                mu_y = self.pix_coords[k][i, 1]
                # star's flux normalized to brightest star
                A = 10**((
                    np.min(self.stars.Tmag.values)
                    - self.stars.Tmag.values[i]
                    )/2.5)
                # integrate PSF in each pixel
                this_flux = 0
                for j in range(len(all_ap_pixels[k])):
                    this_pixel = all_ap_pixels[k][j]
                    this_flux += dblquad(
                        Gauss2D,
                        this_pixel[1]-0.5,
                        this_pixel[1]+0.5,
                        this_pixel[0]-0.5,
                        this_pixel[0]+0.5,
                        args=(mu_x, mu_y, 0.75, A))[0]
                rel_flux_per_aperture[k, i] = this_flux
            # calculate flux ratios for this aperture
            flux_ratio_per_aperture[k, :] = (
                rel_flux_per_aperture[k, :]
                / np.sum(rel_flux_per_aperture[k])
                )

        # take average of flux ratios across all apertures and append
        # to stars dataframe
        flux_ratios = np.mean(flux_ratio_per_aperture, axis=0)
        self.stars["fluxratio"] = flux_ratios
        # calculate transit depth of each star given input transit depth
        tdepths = np.zeros(len(self.stars))
        for i in range(len(flux_ratios)):
            if flux_ratios[i] != 0:
                tdepths[i] = 1-(flux_ratios[i]-tdepth)/flux_ratios[i]
        tdepths[tdepths > 1] = 0
        self.stars["tdepth"] = tdepths

        # check target and possible NFPs for missing properties
        filtered_stars = self.stars[self.stars["tdepth"] > 0]
        for i, ID in enumerate(filtered_stars["ID"].values):
            M_s = filtered_stars["mass"].values[i]
            R_s = filtered_stars["rad"].values[i]
            Teff = filtered_stars["Teff"].values[i]
            Tmag = filtered_stars["Tmag"].values[i]
            plx = filtered_stars["plx"].values[i]
            if i == 0:
                if (np.isnan(M_s) or np.isnan(R_s)
                    or np.isnan(Teff) or np.isnan(plx)):
                    print(
                        "WARNING: " + str(ID)
                        + " is missing stellar properties required "
                        + "for validation."
                        + "Please ensure a stellar "
                        + "mass (in M_Sun), radius (in R_Sun), "
                        + "Teff (in K), and plx (in mas) "
                        + "are provided in the .stars dataframe."
                        )
            else:
                if (np.isnan(M_s) or np.isnan(R_s)
                    or np.isnan(Teff)):
                    print(
                        "WARNING: " + str(ID)
                        + " is missing stellar properties. "
                        + "If a mass (in M_Sun), "
                        + "radius (in R_Sun), and/or Teff (in K) "
                        + "are not added to the .stars dataframe, "
                        + "Solar values will be assumed."
                        )
        return

    def remove_star(self, drop_stars: np.ndarray):
        """
        Drops stars from .stars dataframe so that they are
        excluded from validation analysis.
        Args:
            drop_stars (numpy array): Array of (int) TIC IDs
                                      for stars to drop.
        """
        self.stars = self.stars[~self.stars["ID"].isin(drop_stars)]
        return

    def calc_probs(self, time: np.ndarray, flux_0: np.ndarray,
                   flux_err_0: float, P_orb: float,
                   contrast_curve_file: str = None, band: str = "TESS",
                   N: int = 1000000, parallel: bool = False,
                   verbose: int = 1):
        """
        Calculates the relative probability of each scenario.
        Args:
            time (numpy array): Time of each data point
                                [days from transit midpoint].
            flux_0 (numpy array): Normalized flux of each data point.
            flux_err_0 (float): Uncertainty of flux.
            P_orb (float): Orbital period [days].
            contrast_curve_file (str): Path to contrast curve text file.
                                       File should contain column with
                                       separations (in arcsec) followed
                                       by column with Delta_mags.
            band (str): Photometric band of contrast curve. Options are
                        TESS, Vis, J, H, and K.
            N (int): Number of draws for MC.
            parallel (bool): Whether or not to simulate light curves
                             in parallel.
            verbose (int): 1 to print progress, 0 to print nothing.
        """
        # construct a new dataframe that gives the values of lnL, best
        # fit parameters, lnprior, and relative probability of
        # each scenario considered
        filtered_stars = self.stars[self.stars["tdepth"] > 0]
        N_scenarios = 3*len(filtered_stars) + 12
        targets = np.zeros(N_scenarios, dtype=np.dtype("i8"))
        star_num = np.zeros(N_scenarios, dtype=np.dtype("i8"))
        scenarios = np.zeros(N_scenarios, dtype=np.dtype('U6'))
        best_M_host = np.zeros(N_scenarios)
        best_R_host = np.zeros(N_scenarios)
        best_u1 = np.zeros(N_scenarios)
        best_u2 = np.zeros(N_scenarios)
        best_P_orb = np.zeros(N_scenarios)
        best_i = np.zeros(N_scenarios)
        best_R_p = np.zeros(N_scenarios)
        best_ecc = np.zeros(N_scenarios)
        best_argp = np.zeros(N_scenarios)
        best_M_EB = np.zeros(N_scenarios)
        best_R_EB = np.zeros(N_scenarios)
        best_fluxratio_EB = np.zeros(N_scenarios)
        best_fluxratio_comp = np.zeros(N_scenarios)
        lnZ = np.zeros(N_scenarios)

        for i, ID in enumerate(filtered_stars["ID"].values):
            # subtract flux from other stars in the aperture
            flux, flux_err = renorm_flux(
                flux_0, flux_err_0, filtered_stars["fluxratio"].values[i]
                )

            M_s = filtered_stars["mass"].values[i]
            R_s = filtered_stars["rad"].values[i]
            Teff = filtered_stars["Teff"].values[i]
            Tmag = filtered_stars["Tmag"].values[i]
            Jmag = filtered_stars["Jmag"].values[i]
            Hmag = filtered_stars["Hmag"].values[i]
            Kmag = filtered_stars["Kmag"].values[i]
            plx = filtered_stars["plx"].values[i]
            Z = 0.0
            ra = filtered_stars["ra"].values[i]
            dec = filtered_stars["dec"].values[i]

            # get url to TRILEGAL results
            output_url = self.trilegal_url

            # target star
            if i == 0:

                # check to see if there are any missing stellar
                # parameters and, if there are, ask for input
                if (np.isnan(M_s) or np.isnan(R_s)
                    or np.isnan(Teff) or np.isnan(plx)):
                    print(
                        "Insufficient information to validate "
                        + str(ID)
                        + ". Please ensure a stellar "
                        + "mass (in M_Sun), radius (in R_Sun), "
                        + "Teff (in K), and plx (in mas) "
                        + "are provided in the .stars dataframe."
                        )
                    break

                else:
                    if verbose == 1:
                        print(
                            "Calculating TP, EB, and EBx2P scenario "
                            + "probabilities for " + str(ID) + "."
                            )

                    res = lnZ_TTP(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        N, parallel
                        )
                    j = 0
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "TP"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]

                    res, res_twin = lnZ_TEB(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        N, parallel
                        )
                    j = 1
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "EB"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]
                    j = 2
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "EBx2P"
                    best_M_host[j] = res_twin["M_s"]
                    best_R_host[j] = res_twin["R_s"]
                    best_u1[j] = res_twin["u1"]
                    best_u2[j] = res_twin["u2"]
                    best_P_orb[j] = res_twin["P_orb"]
                    best_i[j] = res_twin["inc"]
                    best_R_p[j] = res_twin["R_p"]
                    best_ecc[j] = res_twin["ecc"]
                    best_argp[j] = res_twin["argp"]
                    best_M_EB[j] = res_twin["M_EB"]
                    best_R_EB[j] = res_twin["R_EB"]
                    best_fluxratio_EB[j] = res_twin["fluxratio_EB"]
                    best_fluxratio_comp[j] = res_twin["fluxratio_comp"]
                    lnZ[j] = res_twin["lnZ"]

                    if verbose == 1:
                        print(
                            "Calculating PTP, PEB, and PEBx2P scenario "
                            + "probabilities for " + str(ID) + "."
                            )

                    res = lnZ_PTP(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        plx, contrast_curve_file,
                        band,
                        N, parallel
                        )
                    j = 3
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "PTP"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]

                    res, res_twin = lnZ_PEB(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        plx, contrast_curve_file,
                        band,
                        N, parallel
                        )
                    j = 4
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "PEB"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]
                    j = 5
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "PEBx2P"
                    best_M_host[j] = res_twin["M_s"]
                    best_R_host[j] = res_twin["R_s"]
                    best_u1[j] = res_twin["u1"]
                    best_u2[j] = res_twin["u2"]
                    best_P_orb[j] = res_twin["P_orb"]
                    best_i[j] = res_twin["inc"]
                    best_R_p[j] = res_twin["R_p"]
                    best_ecc[j] = res_twin["ecc"]
                    best_argp[j] = res_twin["argp"]
                    best_M_EB[j] = res_twin["M_EB"]
                    best_R_EB[j] = res_twin["R_EB"]
                    best_fluxratio_EB[j] = res_twin["fluxratio_EB"]
                    best_fluxratio_comp[j] = res_twin["fluxratio_comp"]
                    lnZ[j] = res_twin["lnZ"]

                    if verbose == 1:
                        print(
                            "Calculating STP, SEB, and SEBx2P scenario "
                            + "probabilities for " + str(ID) + "."
                            )

                    res = lnZ_STP(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        plx, contrast_curve_file,
                        band,
                        N, parallel
                        )
                    j = 6
                    targets[j] = ID
                    star_num[j] = 2
                    scenarios[j] = "STP"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]

                    res, res_twin = lnZ_SEB(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        plx, contrast_curve_file,
                        band,
                        N, parallel
                        )
                    j = 7
                    targets[j] = ID
                    star_num[j] = 2
                    scenarios[j] = "SEB"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]
                    j = 8
                    targets[j] = ID
                    star_num[j] = 2
                    scenarios[j] = "SEBx2P"
                    best_M_host[j] = res_twin["M_s"]
                    best_R_host[j] = res_twin["R_s"]
                    best_u1[j] = res_twin["u1"]
                    best_u2[j] = res_twin["u2"]
                    best_P_orb[j] = res_twin["P_orb"]
                    best_i[j] = res_twin["inc"]
                    best_R_p[j] = res_twin["R_p"]
                    best_ecc[j] = res_twin["ecc"]
                    best_argp[j] = res_twin["argp"]
                    best_M_EB[j] = res_twin["M_EB"]
                    best_R_EB[j] = res_twin["R_EB"]
                    best_fluxratio_EB[j] = res_twin["fluxratio_EB"]
                    best_fluxratio_comp[j] = res_twin["fluxratio_comp"]
                    lnZ[j] = res_twin["lnZ"]

                    if verbose == 1:
                        print(
                            "Calculating DTP, DEB, and DEBx2P scenario "
                            + "probabilities for " + str(ID) + "."
                            )

                    res = lnZ_DTP(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        Tmag, Jmag, Hmag, Kmag,
                        output_url,
                        contrast_curve_file, band,
                        N, parallel
                        )
                    j = 9
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "DTP"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]

                    res, res_twin = lnZ_DEB(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff, Z,
                        Tmag, Jmag, Hmag, Kmag,
                        output_url,
                        contrast_curve_file, band,
                        N, parallel
                        )
                    j = 10
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "DEB"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]
                    j = 11
                    targets[j] = ID
                    star_num[j] = 1
                    scenarios[j] = "DEBx2P"
                    best_M_host[j] = res_twin["M_s"]
                    best_R_host[j] = res_twin["R_s"]
                    best_u1[j] = res_twin["u1"]
                    best_u2[j] = res_twin["u2"]
                    best_P_orb[j] = res_twin["P_orb"]
                    best_i[j] = res_twin["inc"]
                    best_R_p[j] = res_twin["R_p"]
                    best_ecc[j] = res_twin["ecc"]
                    best_argp[j] = res_twin["argp"]
                    best_M_EB[j] = res_twin["M_EB"]
                    best_R_EB[j] = res_twin["R_EB"]
                    best_fluxratio_EB[j] = res_twin["fluxratio_EB"]
                    best_fluxratio_comp[j] = res_twin["fluxratio_comp"]
                    lnZ[j] = res_twin["lnZ"]

                    if verbose == 1:
                        print(
                            "Calculating BTP, BEB, and BEBx2P scenario "
                            + "probabilities for " + str(ID) + "."
                            )

                    res = lnZ_BTP(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff,
                        Tmag, Jmag, Hmag, Kmag,
                        output_url,
                        contrast_curve_file, band,
                        N, parallel
                        )
                    j = 12
                    targets[j] = ID
                    star_num[j] = 2
                    scenarios[j] = "BTP"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]

                    res, res_twin = lnZ_BEB(
                        time, flux, flux_err, P_orb,
                        M_s, R_s, Teff,
                        Tmag, Jmag, Hmag, Kmag,
                        output_url,
                        contrast_curve_file, band,
                        N, parallel
                        )
                    j = 13
                    targets[j] = ID
                    star_num[j] = 2
                    scenarios[j] = "BEB"
                    best_M_host[j] = res["M_s"]
                    best_R_host[j] = res["R_s"]
                    best_u1[j] = res["u1"]
                    best_u2[j] = res["u2"]
                    best_P_orb[j] = res["P_orb"]
                    best_i[j] = res["inc"]
                    best_R_p[j] = res["R_p"]
                    best_ecc[j] = res["ecc"]
                    best_argp[j] = res["argp"]
                    best_M_EB[j] = res["M_EB"]
                    best_R_EB[j] = res["R_EB"]
                    best_fluxratio_EB[j] = res["fluxratio_EB"]
                    best_fluxratio_comp[j] = res["fluxratio_comp"]
                    lnZ[j] = res["lnZ"]
                    j = 14
                    targets[j] = ID
                    star_num[j] = 2
                    scenarios[j] = "BEBx2P"
                    best_M_host[j] = res_twin["M_s"]
                    best_R_host[j] = res_twin["R_s"]
                    best_u1[j] = res_twin["u1"]
                    best_u2[j] = res_twin["u2"]
                    best_P_orb[j] = res_twin["P_orb"]
                    best_i[j] = res_twin["inc"]
                    best_R_p[j] = res_twin["R_p"]
                    best_ecc[j] = res_twin["ecc"]
                    best_argp[j] = res_twin["argp"]
                    best_M_EB[j] = res_twin["M_EB"]
                    best_R_EB[j] = res_twin["R_EB"]
                    best_fluxratio_EB[j] = res_twin["fluxratio_EB"]
                    best_fluxratio_comp[j] = res_twin["fluxratio_comp"]
                    lnZ[j] = res_twin["lnZ"]

            # nearby stars
            else:
                # if a property is unknown, assume solar value
                if np.isnan(Teff):
                    Teff = 5777
                if np.isnan(M_s):
                    M_s = 1.0
                if np.isnan(R_s):
                    R_s = 1.0
                if verbose == 1:
                    print(
                        "Calculating NTP, NEB, and NEB2xP scenario "
                        + "probabilities for " + str(ID) + "."
                        )

                res = lnZ_TTP(
                    time, flux, flux_err, P_orb,
                    M_s, R_s, Teff, Z,
                    N, parallel
                    )
                j = 15 + 3*(i-1)
                targets[j] = ID
                star_num[j] = 1
                scenarios[j] = "NTP"
                best_M_host[j] = res["M_s"]
                best_R_host[j] = res["R_s"]
                best_u1[j] = res["u1"]
                best_u2[j] = res["u2"]
                best_P_orb[j] = res["P_orb"]
                best_i[j] = res["inc"]
                best_R_p[j] = res["R_p"]
                best_ecc[j] = res["ecc"]
                best_argp[j] = res["argp"]
                best_M_EB[j] = res["M_EB"]
                best_R_EB[j] = res["R_EB"]
                best_fluxratio_EB[j] = res["fluxratio_EB"]
                best_fluxratio_comp[j] = res["fluxratio_comp"]
                lnZ[j] = res["lnZ"]

                res, res_twin = lnZ_TEB(
                    time, flux, flux_err, P_orb,
                    M_s, R_s, Teff, Z,
                    N, parallel
                    )
                j = 16 + 3*(i-1)
                targets[j] = ID
                star_num[j] = 1
                scenarios[j] = "NEB"
                best_M_host[j] = res["M_s"]
                best_R_host[j] = res["R_s"]
                best_u1[j] = res["u1"]
                best_u2[j] = res["u2"]
                best_P_orb[j] = res["P_orb"]
                best_i[j] = res["inc"]
                best_R_p[j] = res["R_p"]
                best_ecc[j] = res["ecc"]
                best_argp[j] = res["argp"]
                best_M_EB[j] = res["M_EB"]
                best_R_EB[j] = res["R_EB"]
                best_fluxratio_EB[j] = res["fluxratio_EB"]
                best_fluxratio_comp[j] = res["fluxratio_comp"]
                lnZ[j] = res["lnZ"]
                j = 17 + 3*(i-1)
                targets[j] = ID
                star_num[j] = 1
                scenarios[j] = "NEBx2P"
                best_M_host[j] = res_twin["M_s"]
                best_R_host[j] = res_twin["R_s"]
                best_u1[j] = res_twin["u1"]
                best_u2[j] = res_twin["u2"]
                best_P_orb[j] = res_twin["P_orb"]
                best_i[j] = res_twin["inc"]
                best_R_p[j] = res_twin["R_p"]
                best_ecc[j] = res_twin["ecc"]
                best_argp[j] = res_twin["argp"]
                best_M_EB[j] = res_twin["M_EB"]
                best_R_EB[j] = res_twin["R_EB"]
                best_fluxratio_EB[j] = res_twin["fluxratio_EB"]
                best_fluxratio_comp[j] = res_twin["fluxratio_comp"]
                lnZ[j] = res_twin["lnZ"]

        # calculate the relative probability of each scenario
        relative_probs = np.zeros(N_scenarios)
        for i in range(N_scenarios):
            relative_probs[i] = (np.exp(lnZ[i])) / np.sum(np.exp(lnZ))

        # now save all of the arrays as a dataframe
        prob_df = DataFrame({
            "ID": targets, "scenario": scenarios,
            "M_s": best_M_host, "R_s": best_R_host,
            "P_orb": best_P_orb, "inc": best_i,
            "ecc": best_ecc, "w": best_argp,
            "R_p": best_R_p,
            "M_EB": best_M_EB, "R_EB": best_R_EB,
            "prob": relative_probs
            })
        self.probs = prob_df
        self.star_num = star_num
        self.u1 = best_u1
        self.u2 = best_u2
        self.fluxratio_EB = best_fluxratio_EB
        self.fluxratio_comp = best_fluxratio_comp

        # calculate the FPP and EBP
        self.FPP = 1-(prob_df.prob[0]+prob_df.prob[3]+prob_df.prob[9])
        if len(prob_df.prob) > 15:
            self.NFPP = np.sum(prob_df.prob[15:])
        else:
            self.NFPP = 0.0

        return

    def plot_fits(self, time: np.ndarray,
                  flux_0: np.ndarray, flux_err_0: float,
                  save: bool = False, fname: str = None):
        """
        Plots light curve for best fit instance of each scenario.
        Args:
            time (numpy array): Time of each data point
                                [days from transit midpoint].
            flux_0 (numpy array): Normalized flux of each data point.
            flux_err_0 (float): Uncertainty of flux.
            save (bool): Whether or not to save plot as pdf.
            fname (str): File name of pdf.
        """
        scenario_idx = self.probs[self.probs["ID"] != 0].index.values
        df = self.probs[self.probs["ID"] != 0]
        star_num = self.star_num[self.probs["ID"] != 0]
        u1s = self.u1[self.probs["ID"] != 0]
        u2s = self.u2[self.probs["ID"] != 0]
        fluxratios_EB = self.fluxratio_EB[self.probs["ID"] != 0]
        fluxratios_comp = self.fluxratio_comp[self.probs["ID"] != 0]

        model_time = np.linspace(min(time), max(time), 100)

        f, ax = plt.subplots(
            len(df)//3, 3, figsize=(12, len(df)//3*4), sharex=True
            )
        for i in range(len(df)//3):
            for j in range(3):
                if i == 0:
                    k = j
                else:
                    k = 3*i+j
                # subtract flux from other stars in the aperture
                idx = np.argwhere(
                    self.stars["ID"].values == str(df["ID"].values[k])
                    )[0, 0]
                flux, flux_err = renorm_flux(
                    flux_0, flux_err_0, self.stars["fluxratio"].values[idx]
                    )
                # all TPs
                if j == 0:
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    a = (
                        (G*df["M_s"].values[k]*Msun)/(4*pi**2)
                        * (df['P_orb'].values[k]*86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    best_model = simulate_TP_transit(
                        model_time,
                        df['R_p'].values[k], df['P_orb'].values[k],
                        df['inc'].values[k], a, df["R_s"].values[k],
                        u1, u2,
                        df["ecc"].values[k], df["w"].values[k],
                        fluxratios_comp[k], comp
                        )
                # all small EBs
                elif j == 1:
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    mass = df["M_s"].values[k] + df["M_EB"].values[k]
                    a = (
                        (G*mass*Msun)/(4*pi**2)
                        * (df['P_orb'].values[k]*86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    best_model = simulate_EB_transit(
                        model_time,
                        df["R_EB"].values[k], fluxratios_EB[k],
                        df['P_orb'].values[k], df['inc'].values[k],
                        a, df["R_s"].values[k], u1, u2,
                        df["ecc"].values[k], df["w"].values[k],
                        fluxratios_comp[k], comp
                        )[0]
                # all twin EBs
                elif j == 2:
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    mass = df["M_s"].values[k] + df["M_EB"].values[k]
                    a = (
                        (G*mass*Msun)/(4*pi**2)
                        * (df['P_orb'].values[k]*86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    best_model = simulate_EB_transit(
                        model_time,
                        df["R_EB"].values[k], fluxratios_EB[k],
                        df['P_orb'].values[k], df['inc'].values[k],
                        a, df["R_s"].values[k], u1, u2,
                        df["ecc"].values[k], df["w"].values[k],
                        fluxratios_comp[k], comp
                        )[0]

                y_formatter = ticker.ScalarFormatter(useOffset=False)
                ax[i, j].yaxis.set_major_formatter(y_formatter)
                ax[i, j].errorbar(
                    time, flux, flux_err, fmt=".",
                    color="blue", alpha=0.1, zorder=0,
                    rasterized=True
                    )
                ax[i, j].plot(
                    model_time, best_model, "k-", lw=5, zorder=2
                    )
                ax[i, j].set_ylabel("normalized flux", fontsize=12)
                ax[i, j].annotate(
                    str(df["ID"].values[k]), xy=(0.05, 0.92),
                    xycoords="axes fraction", fontsize=12
                    )
                ax[i, j].annotate(
                    str(df["scenario"].values[k]), xy=(0.05, 0.05),
                    xycoords="axes fraction", fontsize=12
                    )
        ax[len(df)//3-1, 0].set_xlabel(
            "days from transit center", fontsize=12
            )
        ax[len(df)//3-1, 1].set_xlabel(
            "days from transit center", fontsize=12
            )
        ax[len(df)//3-1, 2].set_xlabel(
            "days from transit center", fontsize=12
            )
        if save is False:
            plt.tight_layout()
            plt.show()
        elif (save is True) & (fname is None):
            plt.tight_layout()
            target_star = self.stars.ID.values[0]
            plt.savefig("TIC"+str(target_star)+"_fits.pdf")
        else:
            plt.tight_layout()
            plt.savefig(fname+".pdf")
        return
