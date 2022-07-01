.. _install:

Installation
============

You can install the most recently released version of this tool via PyPI::

    $ pip install triceratops

If you are having trouble getting TRICERATOPS working on your machine, I recommend installing it in a fresh conda environment. You can download the latest distribution of anaconda `here <https://www.anaconda.com/distribution/>`_. After doing so, run the following in terminal::

    $ conda create -n myenv python=3.8
    $ conda activate myenv
    (myenv) $ pip install triceratops jupyterlab

You can replace ``myenv`` with an environment name of your choice. To exit this environment, run::

    (myenv) $ conda deactivate

To delete this environment, run::

    $ conda remove --name myenv --all