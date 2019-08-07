#!/usr/bin/env python3
from setuptools import *

setup(
    name='Korali',
    version=1.00,
    author='G. Arampatzis, S. Martin, D. Waelchli',
    author_email='martiser@ethz.ch',
    description='High Performance Library for Uncertainty Quantification',
    long_description='',
    packages = ['korali', 'korali.plotter', 'korali.cxx'],
    package_dir = {'korali' : '.',
                   'korali.plotter' : './python/plotter', 
                   'korali.cxx' : './python/cxx',
                  }, 
    include_package_data=True,
    package_data={'korali': installFiles },
    install_requires=['pybind11', 'numpy', 'matplotlib']
)
