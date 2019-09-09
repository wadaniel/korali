#!/usr/bin/env python3
import os
from setuptools import *

print('[Korali] Building installation setup...')

sourceDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 
installFiles = ['libkorali.so']
for dirpath, subdirs, files in os.walk(sourceDir):
 for x in files:
  if (x.endswith(".hpp") or (x.endswith(".h")) or (x.endswith(".json")) or x.endswith(".py")):
   relDir = os.path.relpath(dirpath, sourceDir)
   relFile = os.path.join(relDir, x)
   installFiles.append(relFile)

setup(
    name='Korali',
    version=1.00,
    author='G. Arampatzis, S. Martin, D. Waelchli',
    author_email='martiser@ethz.ch',
    description='High Performance Library for Uncertainty Quantification',
    long_description='',
    packages = ['korali', 'korali.auxiliars', 'korali.plotter', 'korali.cxx'],
    package_dir = {'korali' : '.',
                   'korali.auxiliars' : './python/auxiliars', 
                   'korali.plotter' : './python/plotter', 
                   'korali.cxx' : './python/cxx',
                  }, 
    include_package_data=True,
    package_data={'korali': installFiles },
    install_requires=['pybind11', 'numpy', 'matplotlib']
)
