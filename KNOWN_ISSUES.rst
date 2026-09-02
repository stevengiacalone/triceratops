Known issues
============

Status of ``pip install triceratops`` and the test suite across Python
versions, from the CI matrix in ``.github/workflows/tests.yml``. Update
this file as issues are found and fixed.

Summary
-------

``pip install triceratops`` and the full test suite pass on **Python
3.8 through 3.13**. The end-to-end tutorial smoke test
(``tests/manual/tutorial_smoke.py``, run as a non-blocking CI job)
passes on 3.10 and 3.12.

Open
----

*None currently.*

Open
----

**Validate the Gaia DR3 background-population default.** As of this
change the blended-star (DTP/DEB/BTP/BEB) scenarios draw their
field-star population from real Gaia DR3 sources by default rather than
a TRILEGAL simulation (``background_population_source``).

So far:

- The Gaia population is physically sensible and its star count scales
  correctly with galactic latitude (WASP-156 field: 275 stars at
  b = -55 deg, 806 at +24 deg, 24459 at +2 deg).
- End to end on WASP-156b (TOI 465.01), N = 1e6: FPP ~ 0.0005,
  NFPP = 0, i.e. still a validated planet. The blended-star scenarios
  contribute only ~0.001 to the FPP for this isolated, high-latitude
  target, so the source choice barely matters here; it will matter much
  more for crowded / low-latitude targets.

Still to do -- a direct Gaia-vs-TRILEGAL comparison on the paper's
benchmark sample (known planets + known NFPs). TRILEGAL has been
unreachable throughout development (its ``stev.oapd.inaf.it`` server
returns SSL / connection errors), so
``background_population_source="trilegal"`` currently just skips the
D/B scenarios. Run ``tests/manual/background_source_comparison.py
--fpp`` on that sample once TRILEGAL is available and confirm no known
planet's classification degrades and no known NFP gets validated.

Known limitations of the Gaia population:

- shallower than TRILEGAL in crowded / low galactic latitude fields
  (Gaia completeness drops with crowding), which biases the D/B FPP
  contribution low there. A ``gaiaunlimited`` selection-function
  correction plus a faint-tail luminosity-function extrapolation would
  address this.
- all field stars are treated as dwarfs (main-sequence Teff/logg from
  BP-RP); the giant population TRILEGAL includes is approximated.
- metallicity is assumed solar (only used for a coarse limb-darkening
  grid lookup).

Follow-up ideas
---------------

- ``pytransit`` is pinned to 2.2, which predates NumPy 2 / SciPy 1.14
  and relies on ``pkg_resources``. ``triceratops`` works around this
  (``triceratops.likelihoods`` shims the removed NumPy/SciPy names;
  ``setup.py`` holds NumPy/SciPy back where wheels exist and requires
  ``setuptools<81`` on Python >= 3.12). A cleaner long-term fix is to
  move to a ``pytransit`` release that supports NumPy 2 (>= 2.5), which
  would touch ``triceratops/likelihoods.py`` and
  ``triceratops/marginal_likelihoods.py``.
- When TRILEGAL is unreachable, ``calc_probs`` skips the background-star
  scenarios (DTP/DEB/DEBx2P/BTP/BEB/BEBx2P) rather than failing. Shipping
  or caching a representative TRILEGAL population would let those
  scenarios still be evaluated during an outage.

Fixed
-----

- **Python 3.12 / 3.13 install** (``ModuleNotFoundError: No module named
  'pkg_resources'``). ``pytransit==2.2`` imports ``pkg_resources``,
  which is not bundled with Python >= 3.12 and was removed from
  ``setuptools`` 81. Fixed by requiring
  ``setuptools<81; python_version >= "3.12"``.

- **Python 3.10+ install** (``numpy.NaN`` / ``scipy.integrate.trapz``
  removed under NumPy 2 / SciPy 1.14 broke the ``pytransit==2.2``
  import). Fixed by the version-split NumPy/SciPy pins in ``setup.py``
  and the shims in ``triceratops/likelihoods.py`` (``np.int``,
  ``np.trapz``, ``np.NaN``, ``np.Inf``, ``scipy.integrate.trapz``).

- **calc_probs crash when TRILEGAL is unavailable**. ``save_trilegal``
  returned a non-path sentinel and printed that the background-star
  scenarios would be ignored, but ``calc_probs`` still ran them, giving
  ``read_csv(0.0)`` -> ``ValueError``. ``calc_probs`` now detects the
  failed query and skips DTP/DEB/DEBx2P/BTP/BEB/BEBx2P (like
  ``drop_scenario``); FPP/NFPP are returned without them.
