from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "triceratops",
      version = '1.1.0',
      description = "Statistical Validation of Transiting Planet Candidates",
      long_description = readme(),
      author = "Steven Giacalone",
      author_email = "steven_giacalone@berkeley.edu",
      url = "https://github.com/stevengiacalone/triceratops",
      packages = find_packages(),
      package_data = {'triceratops': ['data/*']},
      python_requires='>=3.8',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
   	'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=[# pytransit 2.2 predates NumPy 2 / SciPy 1.14 and
                        # uses names removed there. On Pythons where a
                        # <2 NumPy is still available we hold it back; on
                        # 3.13+ (NumPy 2 only) triceratops.likelihoods
                        # shims the removed names instead.
                        'numpy>=1.18.1,<2; python_version < "3.13"',
                        'numpy>=2.1; python_version >= "3.13"',
                        'scipy>=1.1.0,<1.14; python_version < "3.13"',
                        'scipy>=1.14; python_version >= "3.13"',
                        # pytransit 2.2 imports pkg_resources, which is no
                        # longer bundled with Python >= 3.12 and was
                        # dropped from setuptools 81
                        'setuptools<81; python_version >= "3.12"',
                        'pandas>=0.23.4', 'matplotlib>=3.5.1',
                        'astropy>=4.0', 'astroquery>=0.4.6', 'pytransit==2.2',
                        'mechanicalsoup>=0.12.0', 'emcee>=3.0.2', 'seaborn>=0.11.1',
                        'numba>=0.52.0', 'pyrr>=0.10.3', 'celerite>=0.4.0', 'lightkurve>=2.0.0',
                        'arviz>=0.12.1', 'corner>=2.2.1', 'beautifulsoup4>=4.11.1'],
      zip_safe=False
)
