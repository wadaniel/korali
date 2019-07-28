#!/usr/bin/env python3

import os
import json
import sys

curdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 

from buildSolvers import *
from buildVariables import *

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
buildSolvers(curdir)

# Processing Variables
buildVariables(curdir)

# Creating setup.py
with open('./.setup.py', 'r') as file: setupString = file.read()
installFiles = ['libkorali.so', 'libkorali.a', '../libkorali.so']
for dirpath, subdirs, files in os.walk('.'):
 for x in files:
  if (x.endswith(".hpp") or x.endswith(".py")):
   installFiles.append(os.path.join(dirpath, x))
setupString = setupString.replace('installFiles', str(installFiles))
with open('./setup.py', 'w') as file: file.write(setupString)

