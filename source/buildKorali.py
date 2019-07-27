#!/usr/bin/env python3

import os
import json
import sys
from auxiliar import *
sys.path.append('./solvers')
from buildSolvers import *

# Obtaining list of solvers, problems, and conduits
problemPaths = [x[0] for x in os.walk('./problems')][1:]
conduitPaths = [x[0] for x in os.walk('./conduits')][1:]

# Creating cxx commands.
makefileConfFile = open("./Makefile.conf", "r")
CXXFLAGS=''
for line in makefileConfFile:
  CXXFLAGS += line.replace('=', '=\'', 1).replace('\n', '') + '\'\n'

with open('./python/cxx/cflags.py.base', 'r') as file: cFlagsString = file.read()
with open('./python/cxx/compiler.py.base', 'r') as file: compilerString = file.read()
with open('./python/cxx/libs.py.base', 'r') as file: libString = file.read()
with open('./python/cxx/cflags.py', 'w+') as file: file.write(CXXFLAGS + '\n' + cFlagsString)
with open('./python/cxx/compiler.py', 'w+') as file: file.write(CXXFLAGS + '\n' + compilerString)
with open('./python/cxx/libs.py', 'w+') as file: libString = file.write(CXXFLAGS + '\n' + libString)
 
# Processing Solvers
variableSettingDeclarationsString = buildSolvers()

# Saving new variable.hpp file
with open('./.variable.hpp', 'r') as file: variableHeaderString = file.read()
variableNewHeaderFile = './variable.hpp'
newHeaderString = variableHeaderString.replace('public:', 'public: \n' + variableSettingDeclarationsString + '\n')
with open(variableNewHeaderFile, 'w') as file: file.write(newHeaderString)

# Creating setup.py
with open('./.setup.py', 'r') as file: setupString = file.read()
installFiles = ['libkorali.so', 'libkorali.a', '../libkorali.so']
for dirpath, subdirs, files in os.walk('.'):
 for x in files:
  if x.endswith(".hpp"):
   installFiles.append(os.path.join(dirpath, x))
setupString = setupString.replace('installFiles', str(installFiles))
with open('./setup.py', 'w') as file: file.write(setupString)

