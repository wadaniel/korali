#!/usr/bin/env python3
import os

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 
 
from buildModules import *
from buildDistributions import *
from buildCxx import *
from buildPlotter import *
from buildKorali import *
from buildSetup import *

# Creating cxx commands.
buildCxx(koraliDir)
 
# Creating plotter main.
buildPlotter(koraliDir) 
 
# Processing Modules
buildModules(koraliDir + '/modules')
 
# Processing Distributions
buildDistributions(koraliDir)

# Creating Korali
buildKorali(koraliDir) 

# Creating setup.py
buildSetup(koraliDir) 
