from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "triceratops",
      version = '0.2.1',
      description = "Statistical Validation of TESS Objects of Interest",
      long_description = readme(),
      author = "Steven Giacalone",
      author_email = "steven_giacalone@berkeley.edu",
      url = "https://github.com/stevengiacalone/triceratops",
      packages = find_packages(),
      package_data = {'triceratops': ['data/*']},
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['numpy>=1.18.1','pandas>=0.23.4', 'scipy>=1.1.0', 'matplotlib>=3.0.3',
                        'astropy>=3.1.2', 'astroquery>=0.4', 'batman-package>=2.4.6',
                        'mechanicalsoup>=0.12.0'],
      zip_safe=False
)
