Known issues
============

Status of ``pip install triceratops`` and the test suite across Python
versions, from the CI matrix in ``.github/workflows/tests.yml``. Update
this file as issues are found and fixed.

Summary
-------

===========  ============  =====================
Python       core tests    full install + suite
===========  ============  =====================
3.8          pass          pass
3.9          pass          pass
3.10         pass          pass
3.11         pass          pass
3.12         pass          fix pending verification
3.13         pass          fix pending verification
===========  ============  =====================

"core" installs only numpy/scipy/astropy/pandas/mechanicalsoup/bs4 and
runs the tests not marked ``heavy``. "full" runs ``pip install .`` and
the whole suite. A separate non-blocking "tutorial smoke" job runs
``tests/manual/tutorial_smoke.py`` (the first tutorial example, end to
end, small N).

Open / recently addressed
-------------------------

1. Python 3.12 / 3.13 install: ``No module named 'pkg_resources'``
   .................................................................

   ``pytransit==2.2`` does ``from pkg_resources import resource_filename``
   in ``pytransit/contamination/contamination.py``. ``pkg_resources`` is
   not bundled with Python >= 3.12 and was removed from ``setuptools``
   81, so the import fails::

       File ".../pytransit/contamination/contamination.py", line 34
         from pkg_resources import resource_filename
       ModuleNotFoundError: No module named 'pkg_resources'

   Addressed by adding ``setuptools<81; python_version >= "3.12"`` to
   ``install_requires``. 3.13 may still hit further NumPy 2 issues in
   ``pytransit`` beyond the shims in ``triceratops/likelihoods.py``
   (``np.int``, ``np.trapz``, ``np.NaN``, ``np.Inf``,
   ``scipy.integrate.trapz``); watch the CI log. The durable fix is to
   bump ``pytransit`` to a NumPy-2-compatible release (>= 2.5), which
   would touch ``triceratops/likelihoods.py`` and
   ``triceratops/marginal_likelihoods.py``.

2. calc_probs crashes when TRILEGAL is unavailable
   ..............................................

   When the TRILEGAL query failed, ``save_trilegal`` returned ``0.0``
   and printed that the background-star scenarios would be ignored, but
   ``calc_probs`` still called ``lnZ_DTP(..., trilegal_fname=0.0)``,
   giving ``read_csv(0.0)`` -> ``ValueError: Invalid file path or buffer
   object type: <class 'float'>``.

   Fixed: ``calc_probs`` now detects a failed query
   (``trilegal_fname`` is not a path) and skips DTP/DEB/DEBx2P/BTP/BEB/
   BEBx2P, the same way ``drop_scenario`` does. FPP/NFPP are still
   returned, computed without those scenarios. The misleading "using
   saved stellar populations instead" message was corrected (there is
   no saved-population fallback).

   Follow-up idea: ship a representative TRILEGAL population (or cache
   the last successful one) as a real fallback so the D/B scenarios can
   still be evaluated when the service is down.

Fixed
-----

- Python 3.10 full install (``numpy.NaN`` / ``scipy.integrate.trapz``
  removed under NumPy 2 / SciPy 1.14 broke the ``pytransit==2.2``
  import). Fixed by the version-split pins in ``setup.py`` and the
  shims in ``triceratops/likelihoods.py``.
