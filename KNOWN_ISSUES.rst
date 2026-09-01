Known issues
===========

Status of ``pip install triceratops`` and the test suite across Python
versions, from the CI matrix in ``.github/workflows/tests.yml``. Update
this file as issues are fixed.

Summary
-------

===========  ============  =====================  =========================
Python       core tests    full install + suite   tutorial smoke (E2E)
===========  ============  =====================  =========================
3.8          pass          pass                   not run
3.9          pass          pass                   not run
3.10         pass          pass                   FAIL
3.11         pass          pass                   not run
3.12         pass          FAIL (import)          FAIL
3.13         pass          FAIL (import)          not run
===========  ============  =====================  =========================

"core" installs only numpy/scipy/astropy/pandas/mechanicalsoup/bs4 and
runs the tests not marked ``heavy``. "full" runs ``pip install .`` and
the whole suite. "tutorial smoke" runs
``tests/manual/tutorial_smoke.py`` (the first tutorial example,
end to end, with a small N).

Supported today: **Python 3.8-3.11**. 3.12 and 3.13 install but the
package fails to import; the tutorial smoke test is failing on every
version it runs on.

Open issues
-----------

1. full install fails to import on Python 3.12
   ...............................................

   The ``full`` job installs cleanly but the import check
   (``python -c "import triceratops.triceratops"``) fails. 3.11 uses the
   same dependency pins (``numpy<2``, ``scipy<1.14``) and passes, so this
   is a 3.12-specific runtime incompatibility somewhere in the pinned
   ``pytransit==2.2`` import chain (``pytransit`` / ``numba`` /
   ``llvmlite`` / ``celerite``).

   TODO: get the CI log for ``full (py3.12)`` -> "Show environment"
   step, identify the failing import, and either add a shim in
   ``triceratops/likelihoods.py`` or pin the offending dependency.

2. full install fails to import on Python 3.13
   ...............................................

   3.13 has no ``numpy<2`` wheels, so ``setup.py`` allows
   ``numpy>=2.1`` / ``scipy>=1.14`` there and ``triceratops.likelihoods``
   shims the NumPy/SciPy names ``pytransit==2.2`` needs
   (``np.int``, ``np.trapz``, ``np.NaN``, ``np.Inf``,
   ``scipy.integrate.trapz``). The import still fails, so either
   ``pytransit`` hits more removed names, or ``celerite==0.4.2`` (no
   3.13 wheel) does not build/import.

   TODO: get the CI log for ``full (py3.13)``. The real fix is likely
   to bump ``pytransit`` to a release that supports NumPy 2
   (>= 2.5), which may require changes to ``triceratops/likelihoods.py``
   and ``triceratops/marginal_likelihoods.py``.

3. tutorial smoke test fails end to end
   ......................................

   ``tests/manual/tutorial_smoke.py`` fails on both 3.10 and 3.12.
   ``full (py3.10)`` passes (including the ``heavy`` tests that build a
   ``target`` and run ``calc_depths``), so this is not an install or
   import problem -- it is in the end-to-end run: the ``target()``
   constructor's network calls (TESSCut, the TRILEGAL form submission),
   the TRILEGAL result download inside ``calc_probs``, or the MC itself.

   TRILEGAL in particular is often slow or briefly unavailable, so this
   may be transient. It needs the CI log (or a local run of
   ``python tests/manual/tutorial_smoke.py``) to tell a real bug from a
   flaky external service.

Fixed
-----

- Python 3.10 full install (was: ``numpy.NaN`` / ``scipy.integrate.trapz``
  removed, broke the ``pytransit==2.2`` import). Fixed by the
  version-split pins in ``setup.py`` and the shims in
  ``triceratops/likelihoods.py``.
