.. _calculation:

How it works
============

This page is a work in progress. For more info, see `Giacalone et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G/abstract>`_.

Input light curve and dilution
------------------------------

As of version 1.1.0, ``calc_depths`` and ``calc_probs`` assume by default
that the light curve you pass in has **already been corrected for
dilution** by nearby stars, i.e. that it is normalized so the
out-of-transit flux of the target alone is 1. SPOC ``PDCSAP`` flux (the
default from ``lightkurve``) already meets this convention, as does most
detrended photometry. TRICERATOPS then uses the target light curve as-is
and derives each nearby star's light curve from it.

If instead you pass a raw aperture (``SAP``) light curve that has not
been dilution-corrected, set ``dilution_corrected=False`` in both
``calc_depths`` and ``calc_probs`` and TRICERATOPS will apply the
correction internally. Passing a pre-corrected light curve without this
flag being ``True`` (or a raw light curve with it ``True``) will bias the
inferred transit depth and therefore the FPP.