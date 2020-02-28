triceratops
======

.. image:: https://img.shields.io/badge/GitHub-stevengiacalone%2Ftriceratops-blue.svg?style=flat
    :target: https://github.com/stevengiacalone/triceratops
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/stevengiacalone/triceratops/blob/master/LICENSE
.. image:: http://img.shields.io/badge/arXiv-2002.00691-orange.svg?style=flat
    :target: https://arxiv.org/abs/2002.00691

A tool for validating TESS Objects of Interest.

The paper corresponding to this tool is currently under review. See `Giacalone & Dressing (2020) <https://arxiv.org/abs/2002.00691>`_ for more information.

Installation
-------------

You can install the most recently released version of this tool via PyPI::

    $ pip install triceratops

Or you can clone the repository::

    $ git clone https://github.com/stevengiacalone/triceratops.git
    $ cd triceratops
    $ python setup.py install

Usage
-------------

``triceratops`` can be easily used with jupyter notebook (with Python 3.6 or higher). See the notebook in the examples/ directory for a brief tutorial.

Help
-------------

If you are having trouble getting ``triceratops`` working on your machine, I recommend installing it in a fresh conda environment. You can install the latest distribution of anaconda `here <https://www.anaconda.com/distribution/>`_. After installing anaconda, run the following in terminal::

    $ conda create -n myenv python=3.6
    $ conda activate myenv
    (myenv) $ pip install triceratops notebook

You can replace ``myenv`` with an environment name of your choice. To exit this environment, run::

    (myenv) $ conda deactivate

To delete this environment, run::

    $ conda remove --name myenv --all
