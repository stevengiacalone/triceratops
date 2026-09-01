Changelog
=========

1.1.0
-----

**Behavior change: light-curve dilution handling.**

``calc_probs`` and ``calc_depths`` now assume by default that the input
light curve has already been corrected for dilution by nearby stars
(``dilution_corrected=True``), i.e. that it is normalized so the
out-of-transit target flux is 1 (as in SPOC PDCSAP flux). The target
light curve is used as-is, and each nearby star's light curve is derived
from it. Pass ``dilution_corrected=False`` for a raw aperture (SAP)
light curve, in which case the dilution correction is applied
internally as before.

Previously the code always treated the input as a raw aperture light
curve and applied the target dilution correction itself. Feeding a
pre-corrected (PDCSAP) light curve therefore corrected the target
transit a second time, making it too deep by roughly ``1 / fluxratio``.
FPP and NFPP values will change for targets in crowded fields; results
for isolated targets are unaffected. Use ``dilution_corrected=False``
to reproduce the pre-1.1.0 behavior with a raw aperture light curve.

Other additions on this release:

- ``calc_probs`` accepts a list of contrast curve files (with a matching
  list of filters, or one filter for all). A simulated companion is
  ruled out if any single curve rules it out.
- Missing stellar mass, radius, and Teff in the TIC are estimated from
  the available photometry (Gaia, 2MASS, Johnson V) using the
  Pecaut & Mamajek (2013) dwarf sequence, on ``target`` construction
  (disable with ``estimate_missing_params=False``; re-run manually with
  ``target.estimate_stellar_params``). Evolved stars are identified and
  assigned a nominal 1 M_Sun.
- ``target.remove_nearby_stars()`` drops all stars except the target,
  for use when follow-up has shown the transit to be on-target.
- ``target.get_spoc_apertures()`` now returns one entry per sector
  (``None`` where an aperture could not be retrieved) instead of
  silently returning a shorter list, and ``get_aperture`` reads the
  SPOC optimal-aperture bitmask bit rather than the maximum mask value.
- Added a GitHub Actions test workflow (a fast "core" job on the light
  dependency set and a "full" job with the complete install), a
  ``pyproject.toml`` with pytest configuration, and unit tests for the
  pure helper functions and prior samplers. Tests that need the full
  install are marked ``heavy``.
- Capped ``numpy < 2`` and ``scipy < 1.14``: the pinned ``pytransit``
  2.2 imports ``numpy.NaN`` and ``scipy.integrate.trapz``, both of
  which have since been removed.
