#!/usr/bin/env python3
import os

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 
 
from buildVariables import *
from buildModules import *
from buildModels import *
from buildDistributions import *
from buildCxx import *
from buildPlotter import *
from buildKorali import *
from buildSetup import *

# Creating cxx commands.
buildCxx(koraliDir)
 
# Creating plotter main.
buildPlotter(koraliDir) 
 
# Processing Solvers
buildModules(koraliDir + '/solvers')
 
# Processing Problems
buildModules(koraliDir + '/problems')

# Processing Models
buildModels(koraliDir)

# Processing Conduits
buildModules(koraliDir + '/conduits')

# Processing Distributions
buildDistributions(koraliDir)

# Processing Variables
buildVariables(koraliDir) 

# Creating Korali
buildKorali(koraliDir) 

# Creating setup.py
buildSetup(koraliDir) 
