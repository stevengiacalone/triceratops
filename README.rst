triceratops
===========

.. image:: https://img.shields.io/badge/GitHub-stevengiacalone%2Ftriceratops-blue.svg?style=flat
    :target: https://github.com/stevengiacalone/triceratops
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/stevengiacalone/triceratops/blob/master/LICENSE
.. image:: http://img.shields.io/badge/arXiv-2002.00691-orange.svg?style=flat
    :target: https://arxiv.org/abs/2002.00691

A tool for vetting and validating TESS Objects of Interest.

See `Giacalone et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G/abstract>`_ for more information about this tool.

For a modified version of the code that can simultaneously analyze transits observed in different photometric passbands (i.e., both TESS data and ground-based data), see `this repo <https://github.com/JGB276/TRICERATOPS_v2>`_.

Installation
-------------

You can install the most recently released version of this tool via PyPI::

    $ pip install triceratops


Usage
-------------

``triceratops`` can be easily used with jupyter notebook (with Python 3.6 or higher). See the notebook in the examples/ directory for a brief tutorial or check out the `documentation <https://triceratops.readthedocs.io/en/latest/>`_.

Attribution
-------------
If you use ``triceratops``, please cite both the paper and the code.

Paper citation::

    @ARTICLE{2021AJ....161...24G,
           author = {{Giacalone}, Steven and {Dressing}, Courtney D. and {Jensen}, Eric L.~N. and {Collins}, Karen A. and {Ricker}, George R. and {Vanderspek}, Roland and {Seager}, S. and {Winn}, Joshua N. and {Jenkins}, Jon M. and {Barclay}, Thomas and {Barkaoui}, Khalid and {Cadieux}, Charles and {Charbonneau}, David and {Collins}, Kevin I. and {Conti}, Dennis M. and {Doyon}, Ren{\'e} and {Evans}, Phil and {Ghachoui}, Mourad and {Gillon}, Micha{\"e}l and {Guerrero}, Natalia M. and {Hart}, Rhodes and {Jehin}, Emmanu{\"e}l and {Kielkopf}, John F. and {McLean}, Brian and {Murgas}, Felipe and {Palle}, Enric and {Parviainen}, Hannu and {Pozuelos}, Francisco J. and {Relles}, Howard M. and {Shporer}, Avi and {Socia}, Quentin and {Stockdale}, Chris and {Tan}, Thiam-Guan and {Torres}, Guillermo and {Twicken}, Joseph D. and {Waalkes}, William C. and {Waite}, Ian A.},
            title = "{Vetting of 384 TESS Objects of Interest with TRICERATOPS and Statistical Validation of 12 Planet Candidates}",
          journal = {\aj},
         keywords = {Exoplanet astronomy, Astrostatistics, Planet hosting stars, Exoplanets, 486, 1882, 1242, 498, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
             year = 2021,
            month = jan,
           volume = {161},
           number = {1},
              eid = {24},
            pages = {24},
              doi = {10.3847/1538-3881/abc6af},
    archivePrefix = {arXiv},
           eprint = {2002.00691},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Code citation::

    @MISC{2020ascl.soft02004G,
           author = {{Giacalone}, Steven and {Dressing}, Courtney D.},
            title = "{triceratops: Candidate exoplanet rating tool}",
         keywords = {Software, NASA, TESS},
             year = 2020,
            month = feb,
              eid = {ascl:2002.004},
            pages = {ascl:2002.004},
    archivePrefix = {ascl},
           eprint = {2002.004},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2020ascl.soft02004G},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Help
-------------

If you are having trouble getting ``triceratops`` working on your machine, I recommend installing it in a fresh conda environment. You can download the latest distribution of anaconda `here <https://www.anaconda.com/distribution/>`_. After doing so, run the following in terminal::

    $ conda create -n myenv python=3.8
    $ conda activate myenv
    (myenv) $ pip install triceratops jupyterlab

You can replace ``myenv`` with an environment name of your choice. To exit this environment, run::

    (myenv) $ conda deactivate

To delete this environment, run::

    $ conda remove --name myenv --all
