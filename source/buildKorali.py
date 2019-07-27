#!/usr/bin/env python3

import os
import json
import sys
from auxiliar import *

curdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 

sys.path.append('./solvers')
from buildSolvers import *

sys.path.append('./python/cxx/')
from buildCXX import *

sys.path.append('./python/plotter/')
from buildPlotter import *

# Obtaining list of solvers, problems, and conduits
problemPaths = [x[0] for x in os.walk('./problems')][1:]
conduitPaths = [x[0] for x in os.walk('./conduits')][1:]

# Creating cxx commands.
buildCXX(curdir)
 
# Creating plotter main.
buildPlotter(curdir) 
 
# Processing Solvers
variableSettingDeclarationsString = buildSolvers(curdir)

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
  if (x.endswith(".hpp") or x.endswith(".py")):
   installFiles.append(os.path.join(dirpath, x))
setupString = setupString.replace('installFiles', str(installFiles))
with open('./setup.py', 'w') as file: file.write(setupString)

