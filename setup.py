#!/usr/bin/env python3
import os
from setuptools import *

print('[Korali] Building installation setup...')
with open('docs/VERSION') as f: koraliVer = f.read()

sourceDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/source/') 
koraliFiles = ['libkorali.so', 'Makefile.conf']
for dirpath, subdirs, files in os.walk(sourceDir):
 for x in files:
  if (x.endswith(".hpp") or (x.endswith(".h")) or (x.endswith(".config")) or x.endswith(".py")):
   relDir = os.path.relpath(dirpath, sourceDir)
   relFile = os.path.join(relDir, x)
   koraliFiles.append(relFile)

mdevFiles =  [ 'solver/base.cpp', 'solver/base.hpp', 'solver/base.config', 'solver/README.rst', 'problem/base.cpp', 'problem/base.hpp', 'problem/base.config', 'problem/README.rst' ]

setup(
    name='Korali',
    version=koraliVer,
    author='G. Arampatzis, S. Martin, D. Waelchli',
    author_email='martiser@ethz.ch',
    description='High Performance Framework for Uncertainty Quantification and Optimization',
    url='Webpage: https://www.cse-lab.ethz.ch/korali/',
    packages = ['korali', 'korali.plotter', 'korali.profiler', 'korali.cxx', 'korali.mdev'],
    package_dir = {'korali' : './source/',
                   'korali.plotter' : './tools/plotter', 
                   'korali.profiler' : './tools/profiler', 
                   'korali.cxx' : './tools/cxx',
                   'korali.mdev' : './tools/mdev'
                  }, 
    include_package_data=True,
    package_data={'korali': koraliFiles, 'korali.mdev': mdevFiles },
    install_requires=['pybind11', 'numpy', 'matplotlib'],
    license='GNU General Public License v3.0'
)
