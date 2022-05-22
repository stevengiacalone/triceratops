TRICERATOPS
=====

TRICERATOPS is a code that uses a Bayesian framework to determine if a transiting planet candidate is a real planet or an astrophysical false positive. The code does this by analyzing the transit data and the surrounding field of stars and calculating a false positive probability (the overall probability that a planet candidate is an astrophysical false positive) and a nearby false positive probabilty (the probability that the transit-like event originates from a known nearby source, such as a pair of nearby eclipsing binaries). TRICERATOPS can also fold in follow-up data, such as high resolution imaging, to provide stronger constraints on false positive scenarios.

TRICERATOPS is open source and has a 'public repository on github <https://github.com/stevengiacalone/triceratops>'_. If you have any questions or issues, feel free to reach out there!

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Installation
   How it works
   API documentation
   Citing TRICERATOPS
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   TESS tutorial
   Kepler/K2 tutorial
   Folding in follow-up
