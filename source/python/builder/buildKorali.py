#!/usr/bin/env python3

import os
import json
import sys

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildSolvers import *
from buildVariables import *
from buildCxx import *
from buildPlotter import *
from buildSetup import *

# Obtaining list of solvers, problems, and conduits
problemPaths = [x[0] for x in os.walk('./problems')][1:]
conduitPaths = [x[0] for x in os.walk('./conduits')][1:]

# Creating cxx commands.
buildCxx(koraliDir)
 
# Creating plotter main.
buildPlotter(koraliDir) 
 
# Processing Solvers
buildSolvers(koraliDir)

# Processing Variables
buildVariables(koraliDir) 

# Creating setup.py
buildSetup(koraliDir) 

