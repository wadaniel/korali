#! /usr/bin/env python3

import os, glob
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import sys
sys.path.insert(0, 'source')

class BinaryExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CopyLibrary(build_ext):
    def run(self):
        for ext in self.extensions:
            self.copy_extension(ext)

    def copy_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        koraliLibs = glob.glob(ext.sourcedir + '/source/libkorali.*')

        if (len(koraliLibs) == 0):
            raise ValueError('No Korali libraries found in folder ' + ext.sourcedir)
            
        for file in koraliLibs:
            shutil.copy2(file, extdir)
            
        koraliIncludes = glob.glob(ext.sourcedir + '/include')

        for file in koraliIncludes:
            shutil.copytree(file, extdir + '/korali/include')

setup(
    name='Korali',
    version=1.00,
    author='G. Arampatzis, S. Martin, D. Waelchli',
    author_email='martiser@ethz.ch',
    description='High Performance Library for Uncertainty Quantification',
    long_description='',
    packages = ['korali', 'korali.plotter', 'korali.cxx'],
    package_dir = {'korali' : 'python'},
    ext_modules=[BinaryExtension('libkorali', sourcedir='./')],
    cmdclass=dict(build_ext=CopyLibrary),
    zip_safe=False,
    install_requires=['pybind11', 'numpy', 'matplotlib']
)
