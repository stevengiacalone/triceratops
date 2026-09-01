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
