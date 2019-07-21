#!/usr/bin/env python3

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
        
        koraliLibDirSrc = ext.sourcedir + '/source'
        koraliSharedLibSrc = glob.glob(koraliLibDirSrc + '/libkorali.so')
        koraliStaticLibSrc = glob.glob(koraliLibDirSrc + '/libkorali.a')

        koraliSharedLibDst = extdir
        koraliStaticLibDst = extdir + '/korali/cxx/lib'

        os.makedirs(koraliStaticLibDst)

        if (len(koraliSharedLibSrc) == 0):
            raise ValueError('No libkorali.so found in folder ' + koraliLibDirSrc)

        if (len(koraliStaticLibSrc) == 0):
            raise ValueError('No libkorali.a found in folder ' + koraliLibDirSrc)
            
        shutil.copy2(koraliSharedLibSrc[0], koraliSharedLibDst)
        shutil.copy2(koraliStaticLibSrc[0], koraliStaticLibDst)
        
        # Creating include files directories
        os.makedirs(extdir + '/korali/cxx/include')
        os.makedirs(extdir + '/korali/cxx/include/solvers')
        os.makedirs(extdir + '/korali/cxx/include/problems')
        os.makedirs(extdir + '/korali/cxx/include/conduits')
        
        # Copying base directory includes  
        koraliIncludesDir = ext.sourcedir + '/source/*.hpp'
        koraliIncludesSrc = glob.glob(koraliIncludesDir)
        
        for file in koraliIncludesSrc:
            shutil.copy2(file, extdir + '/korali/cxx/include')
        
        # Copying solver directory includes    
        koraliIncludesDir = ext.sourcedir + '/source/solvers/*.hpp'
        koraliIncludesSrc = glob.glob(koraliIncludesDir)
        
        for file in koraliIncludesSrc:
            shutil.copy2(file, extdir + '/korali/cxx/include/solvers')
        
        # Copying problems directory includes    
        koraliIncludesDir = ext.sourcedir + '/source/problems/*.hpp'
        koraliIncludesSrc = glob.glob(koraliIncludesDir)
        
        for file in koraliIncludesSrc:
            shutil.copy2(file, extdir + '/korali/cxx/include/problems')
            
        # Copying conduits directory includes    
        koraliIncludesDir = ext.sourcedir + '/source/conduits/*.hpp'
        koraliIncludesSrc = glob.glob(koraliIncludesDir)
        
        for file in koraliIncludesSrc:
            shutil.copy2(file, extdir + '/korali/cxx/include/conduits')

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
