#!/usr/bin/env python3
from setuptools import *
from setuptools.command.build_ext import build_ext

class linkLibrary(build_ext):
    def run(self):
    # Creating symlink
     extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
     if (os.path.exists(extdir + '/libkorali.so')): os.remove(extdir + '/libkorali.so')
     os.symlink(extdir + '/korali/cxx/source/libkorali.so', extdir + '/libkorali.so') 

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
    cmdclass=dict(build_ext=linkLibrary),
    include_package_data=True,
    package_data={'korali': installFiles },
    install_requires=['pybind11', 'numpy', 'matplotlib']
)
