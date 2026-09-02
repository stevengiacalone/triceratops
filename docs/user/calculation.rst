.. _calculation:

How it works
============

This page summarizes the method behind TRICERATOPS. For the full
description, the underlying data, and validation tests, see
`Giacalone et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G/abstract>`_.

Overview
--------

TRICERATOPS decides whether a transiting planet candidate (a *TESS
Object of Interest*, or TOI) is a genuine planet or an astrophysical
false positive. It does this with a Bayesian model comparison: for the
target star and every nearby star that could plausibly produce the
signal, it enumerates the ways a transit-like event could arise
(a planet, an eclipsing binary, a blended eclipsing binary, ...),
computes the probability of each, and combines them into two numbers:

- the **false positive probability (FPP)** — the probability that the
  signal is *not* a planet transiting the target star, and
- the **nearby false positive probability (NFPP)** — the probability
  that the signal actually comes from a *resolved* nearby star.

The two are reported together because they answer different questions:
FPP folds in blended scenarios that ground-based seeing cannot resolve,
while NFPP flags candidates whose signal can be chased down with
seeing-limited follow-up photometry. Giacalone et al. (2021) place TOIs
in the FPP–NFPP plane and define:

- **validated planet**: ``FPP < 0.015`` and ``NFPP < 1e-3``
- **likely planet**: ``FPP < 0.5`` and ``NFPP < 1e-3``
- **likely nearby false positive**: ``NFPP > 0.1``

The calculation proceeds in three stages, described below: measuring how
much each star contributes to the photometric aperture, enumerating the
scenarios, and computing scenario probabilities.

Step 1 — the field of stars and their flux ratios
-------------------------------------------------

When a ``target`` object is created, TRICERATOPS queries the TIC for
every star within 10 pixels of the target and records its position,
TESS magnitude, and stellar properties. The user then supplies the
photometric aperture used to extract the light curve in each sector
(``target.get_spoc_apertures()`` fetches the SPOC pipeline aperture
automatically).

Each star's point spread function is modeled as a circular 2D Gaussian
with a standard deviation of 0.75 pixels, normalized to the star's TESS
flux. Integrating each model over the aperture pixels and dividing by
the summed flux of all stars gives the **flux ratio** :math:`X_s` — the
fraction of the aperture's light that comes from star :math:`s`. For a
target observed in several sectors, :math:`X_s` is averaged across
sectors. These values match the SPOC pipeline's own contamination
estimates well (Giacalone et al. 2021, Figure 1) and are stored in the
``fluxratio`` column of ``target.stars``.

Given the observed transit depth :math:`\delta_\mathrm{obs}`, the depth
the signal would need to have if it originated entirely from star
:math:`s` is

.. math::

   \delta_s = \delta_\mathrm{obs} / X_s .

A faint star contributes so little light that :math:`\delta_s` exceeds
1 (a physically impossible eclipse). Those stars are dropped; only
stars with :math:`\delta_s < 1` are carried into the scenario analysis.
This is what ``target.calc_depths()`` computes (the ``tdepth`` column).

Step 2 — the transit scenarios
------------------------------

For the target star, TRICERATOPS considers **fifteen** scenarios,
grouped by what (if anything) is blended with the target.

**No unresolved companion:**

- ``TP`` — transiting planet around the target.
- ``EB`` — eclipsing binary around the target.
- ``EBx2P`` — eclipsing binary at twice the reported period (near-equal
  stars whose primary and secondary eclipses are mistaken for one
  planet transit at half the period).

**Unresolved bound companion** (a physically associated star):

- ``PTP`` / ``PEB`` / ``PEBx2P`` — planet / EB / EBx2P around the
  **primary** (brighter) star.
- ``STP`` / ``SEB`` / ``SEBx2P`` — planet / EB / EBx2P around the
  **secondary** (fainter bound) star.

**Unresolved foreground/background star** (a chance line-of-sight
alignment):

- ``DTP`` / ``DEB`` / ``DEBx2P`` — planet / EB / EBx2P around the
  **target**, with the transit diluted by the extra star.
- ``BTP`` / ``BEB`` / ``BEBx2P`` — planet / EB / EBx2P around the
  **background** star itself.

For every *resolved* nearby star that survived the :math:`\delta_s < 1`
cut, three more scenarios are added:

- ``NTP`` / ``NEB`` / ``NEBx2P`` — planet / EB / EBx2P around that
  nearby star (assumed to have no unresolved companion of its own).

The ``x2P`` ("twin") variants require a stellar mass ratio
:math:`q > 0.95`; the ordinary ``EB`` variants require :math:`q < 0.95`
and a predicted secondary-eclipse depth shallower than
:math:`1.5\times` the light-curve scatter (otherwise the secondary
eclipse would have been noticed).

Step 3 — scenario probabilities
-------------------------------

Bayes' theorem gives the probability of scenario :math:`S_j` after
seeing the data :math:`D`:

.. math::

   p(S_j \mid D) \propto p(S_j)\, p(D \mid S_j),

where :math:`p(S_j)` is the **scenario prior** and :math:`p(D \mid S_j)`
is the **marginal likelihood** (Bayesian evidence) — the likelihood
averaged over the scenario's parameter prior:

.. math::

   p(D \mid S_j) = \int p(\theta_j \mid S_j)\, p(D \mid \theta_j, S_j)\,
   d\theta_j .

The relative probability of each scenario is then

.. math::

   P_j = \frac{p(S_j \mid D)}{\sum_k p(S_k \mid D)} ,

stored in the ``prob`` column of ``target.probs``.

Marginal likelihood
~~~~~~~~~~~~~~~~~~~~~

The integral is evaluated by Monte Carlo (the arithmetic-mean
estimator of Kass & Raftery 1995): draw :math:`N` parameter vectors
:math:`\theta_j^{(n)}` from the parameter prior and average the
likelihood,

.. math::

   p(D \mid S_j) \approx \frac{1}{N} \sum_{n=1}^{N}
   p(D \mid \theta_j^{(n)}, S_j) .

The default :math:`N = 10^6` is chosen so that repeated runs agree to
within a few percent. TRICERATOPS keeps this estimator well-behaved for
long light curves by working in log space (``triceratops._numerics``):
draws whose geometry does not produce a transit contribute zero weight
but are still counted in the denominator, so scenarios that are
geometrically excluded are correctly penalized.

Each draw's likelihood has two factors:

.. math::

   p(D \mid \theta_j^{(n)}, S_j) =
   p(D_\mathrm{tra} \mid \theta_j^{(n)}, S_j) \times w^{(n)} .

**Transit-data term.** A Gaussian likelihood comparing the observed
flux :math:`y_l` to a model light curve :math:`f(t_l \mid \theta_j)`:

.. math::

   p(D_\mathrm{tra} \mid \theta_j^{(n)}, S_j) \propto
   \prod_l \exp\!\left[ -\tfrac{1}{2}
   \left( \frac{y_l - f(t_l \mid \theta_j^{(n)})}{\sigma} \right)^2
   \right] .

Model light curves are generated with an analytic transit/eclipse
model (currently `pytransit <https://github.com/hpparvi/PyTransit>`_;
``batman`` in Giacalone et al. 2021), using quadratic limb-darkening
coefficients selected from the host :math:`T_\mathrm{eff}` and
:math:`\log g` (Claret 2018) and supersampled to the observing cadence.

**Follow-up weight** :math:`w^{(n)}`. This factor down-weights blended
scenarios by the frequency of the star doing the blending:

- For the bound-companion scenarios (``PTP``, ``PEB``, ``STP``,
  ``SEB``, and their twins), :math:`w^{(n)}` is the bound-companion
  frequency from Moe & Di Stefano (2017). For each draw: the magnitude
  difference between the two stars sets, via a **contrast curve**, the
  angular separation beyond which the companion would have been
  resolved; the target parallax and the two masses convert that to a
  maximum orbital period; and that period gives the companion frequency.
- For the background-star scenarios (``DTP``, ``DEB``, ``BTP``,
  ``BEB``, and their twins), :math:`w^{(n)}` is the frequency of a
  chance-aligned foreground/background star, estimated from the
  field-star population in a 0.1 deg\ :sup:`2` field around the target
  (stars brighter than the target are removed), again combined with the
  contrast curve. By default this population is **real Gaia DR3
  sources** down to G = 21, with stellar properties from main-sequence
  relations and a TESS magnitude from the Stassun et al. (2019) G-T
  relation; pass ``background_population_source="trilegal"`` to
  ``target`` to use a
  `TRILEGAL <https://stev.oapd.inaf.it/cgi-bin/trilegal>`_ simulation
  instead. Gaia removes the dependence on the TRILEGAL web service and
  captures the real field, but is shallower in crowded / low-latitude
  fields where TRILEGAL is the better choice.
- :math:`w^{(n)}` is capped at 1, and set to 1 for scenarios with no
  unresolved companion (``TP``, ``EB``, ``EBx2P``, ``NTP``, ``NEB``,
  ``NEBx2P``).

If no contrast curve is supplied, the limiting separation defaults to
2.2 arcsec (Brown et al. 2018). Contrast curves are passed via the
``contrast_curve_file`` argument of ``calc_probs`` (a single file, or a
list of files with a matching list of filters — a companion is ruled
out if *any* curve rules it out).

Scenario prior
~~~~~~~~~~~~~~~

The only scenario prior used is the probability that a transiting
planet or eclipsing binary has the reported orbital period. Both are
modeled as broken power laws over 0.1–50 days:

.. math::

   p(P_\mathrm{orb}) \propto
   \begin{cases}
     P_\mathrm{orb}^{\,1.5}, & 0.1 \le P_\mathrm{orb} \le 10 \ \text{d} \\
     P_\mathrm{orb}^{\,0.0}, & 10 < P_\mathrm{orb} \le 50 \ \text{d}
   \end{cases}
   \quad \text{(planets)}

.. math::

   p(P_\mathrm{orb}) \propto
   \begin{cases}
     P_\mathrm{orb}^{\,5.0}, & 0.1 \le P_\mathrm{orb} \le 0.3 \ \text{d} \\
     P_\mathrm{orb}^{\,0.5}, & 0.3 < P_\mathrm{orb} \le 50 \ \text{d}
   \end{cases}
   \quad \text{(eclipsing binaries)}

Priors that capture the *overall* planet occurrence and stellar
multiplicity rates (which would favor planet scenarios by
10–100\ :math:`\times`) are deliberately **omitted**: in testing they
gave planet scenarios too much of an advantage and caused FPP to be
underestimated.

FPP and NFPP
------------

With the scenario probabilities :math:`P_j` in hand,

.. math::

   \mathrm{FPP} = 1 - (P_\mathrm{TP} + P_\mathrm{PTP} + P_\mathrm{DTP})

.. math::

   \mathrm{NFPP} = \sum \left( P_\mathrm{NTP} + P_\mathrm{NEB}
   + P_\mathrm{NEBx2P} \right) .

The three scenarios subtracted in the FPP are exactly those in which a
genuine planet with the reported period orbits the target star — on its
own (``TP``), as the brighter member of an unresolved bound pair
(``PTP``), or with its transit diluted by an unrelated blended star
(``DTP``). Everything else counts as a false positive. The NFPP sums
the scenarios in which a *resolved* neighbor is the true host.

After ``calc_probs()`` these are available as ``target.FPP`` and
``target.NFPP``.

Parameter priors
----------------

Each scenario marginalizes over a small parameter vector. The
distributions sampled (Giacalone et al. 2021, Section 2.4.2) are:

- **Inclination** :math:`i`: isotropic orbits, :math:`p(i) \propto
  \sin i`.
- **Planet radius** :math:`R_p`: a broken power law over
  0.5–20 :math:`R_\oplus` with breaks at 3 and 6 :math:`R_\oplus`,
  using a steeper suppression of giant planets around M dwarfs
  (:math:`R_p^{-7}` between 3–6 :math:`R_\oplus`) than around FGK
  dwarfs (:math:`R_p^{-4}`). Pass ``flatpriors=True`` to ``calc_probs``
  for a uniform :math:`R_p` prior instead.
- **Short-period mass ratio** :math:`q_\mathrm{short}` (eclipsing
  binaries): a broken power law from Moe & Di Stefano (2017),
  :math:`q^{0.3}` below :math:`q = 0.3` and :math:`q^{-5.0}` above,
  with a twin excess :math:`F_\mathrm{twin} = 0.3`.
- **Long-period mass ratio** :math:`q_\mathrm{long}` (unresolved bound
  companions): :math:`q^{0.3}` below :math:`q = 0.3` and
  :math:`q^{-0.95}` above, with :math:`F_\mathrm{twin} = 0.05`.
- **Field star**: for the diluted/background scenarios, the blended
  star's properties are drawn at random from the field-star population
  (Gaia DR3 or TRILEGAL; see above).

The eccentricity is fixed to zero and the orbital period is fixed to
the reported value (or sampled uniformly between ``P_min`` and
``P_max`` if a range is given). Circular orbits are a good
approximation for the short periods typical of TOIs but become less
accurate for longer periods.

Stellar properties
------------------

Resolved stars use their TIC properties, treated as exact. Stars that
are missing a mass, radius, and/or :math:`T_\mathrm{eff}` in the TIC
are filled in on ``target`` construction
(``target.estimate_stellar_params``): :math:`T_\mathrm{eff}` is
anchored to the best available dereddened color and the mass and radius
follow from the main-sequence (dwarf) sequence of Pecaut & Mamajek
(2013), with a Stefan–Boltzmann radius when a parallax is available.
Stars identified as evolved are given a nominal 1 :math:`M_\odot`.
Which values were estimated is recorded in the ``params_estimated``
column of ``target.stars``.

Unresolved companions and eclipsing-binary members, whose masses come
from the sampled mass ratios, are characterized from mass alone: their
radii and effective temperatures follow spline relations in
:math:`M_\star`–:math:`R_\star` and :math:`M_\star`–:math:`T_\mathrm{eff}`
space (Torres et al. 2010 above :math:`0.63\,M_\odot`, the TESS Cool
Dwarf Catalog below), and their TESS-band flux relative to the host
follows a spline in the :math:`M_\star`–TESS-magnitude plane.
Metallicity is assumed solar throughout.

Input light curve and dilution
------------------------------

As of version 1.1.0, ``calc_depths`` and ``calc_probs`` assume by
default that the light curve you pass in has **already been corrected
for dilution** by nearby stars, i.e. that it is normalized so the
out-of-transit flux of the target alone is 1. SPOC ``PDCSAP`` flux (the
default from ``lightkurve``) already meets this convention, as does
most detrended photometry. TRICERATOPS then uses the target light curve
as-is and derives each nearby star's light curve from it.

If instead you pass a raw aperture (``SAP``) light curve that has not
been dilution-corrected, set ``dilution_corrected=False`` in both
``calc_depths`` and ``calc_probs`` and TRICERATOPS will apply the
correction internally. Passing a pre-corrected light curve without this
flag being ``True`` (or a raw light curve with it ``True``) will bias
the inferred transit depth and therefore the FPP.

Assumptions and limitations
---------------------------

- The reported orbital period, transit depth, and (unless overridden)
  the target's stellar properties are taken as exact.
- Orbits are circular; the analysis degrades for long periods.
- Nearby stars are assumed to have no unresolved companions of their
  own.
- The FPP inherits the planet-radius and orbital-period priors. For
  large-scale occurrence-rate work, consider ``flatpriors=True`` so the
  result does not encode a previous occurrence rate.
- TRICERATOPS does not use the transit's measured centroid or
  pixel-level position; that information is complementary.
- A low NFPP does not by itself validate a planet — it only means the
  signal is unlikely to come from a *resolved* neighbor.
- The default Gaia DR3 field-star population is shallower than a
  TRILEGAL simulation in crowded / low galactic latitude fields, which
  biases the blended-star (D/B) contribution to the FPP low there.
  Field stars are all treated as dwarfs. Use
  ``background_population_source="trilegal"`` for such targets.

Practical recommendations
-------------------------

- **Run the calculation several times** (e.g. 20) and report the mean
  and standard deviation of the FPP; the Monte Carlo integration has a
  few-percent run-to-run scatter.
- Use the **SPOC aperture** (``target.get_spoc_apertures()``) and the
  matching **PDCSAP** light curve so the flux ratios and the dilution
  convention are consistent.
- **Fold in a contrast curve** from high-resolution imaging whenever
  one is available: it directly tightens the blended-companion
  scenarios and can move a "likely planet" across the validation
  threshold.
- If seeing-limited follow-up has shown the transit to be **on-target**,
  call ``target.remove_nearby_stars()`` before ``calc_probs()`` to drop
  the ``NTP``/``NEB`` scenarios (``NFPP`` becomes 0).
- For targets in crowded / low galactic latitude fields, build the
  ``target`` with ``background_population_source="trilegal"`` so the
  blended-star scenarios use a population that stays complete below the
  Gaia limit.
- Use ``drop_scenario`` to exclude specific scenarios that independent
  information has ruled out.
