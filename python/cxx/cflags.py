#! /usr/bin/env python3

VERSION=1.0
PREFIX='/Users/martiser/skorali/korali'
CXX=clang++
CXXFLAGS="-I/Users/martiser/skorali/korali/include -I/Users/martiser/skorali/korali/lib/koraligsl/include/gsl --std=c++14 -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address -Wno-deprecated-declarations -Wall -Wfatal-errors -Wno-c++17-extensions"
LDFLAGS="-L/usr/local/opt/python/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin -lpython3.7m -ldl -framework CoreFoundation"
SHAREDLIB_FLAG=-dynamiclib
USE_MPI=0
DLL_EXTENSION=dylib

import subprocess

def getcflags(extdir):
 print('-O3 -I' + extdir + '/include --std=c++14 -Wno-c++17-extensions `python3 -m pybind11 --includes` `gsl-config --libs`')