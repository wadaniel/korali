#!/usr/bin/env python3
import os

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildSolvers import *
from buildVariables import *
from buildCxx import *
from buildPlotter import *
from buildSetup import *

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

