#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali

e = korali.newExperiment()
e["Console Frequency"] = 500
e["Save Frequency"] = 500

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 1.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["Result Path"] = "_result_run-mcmc"

k = korali.initialize()
k.run(e)

# Testing Results
checkMean(e, 0.0, 0.01)
checkStd(e, 1.0, 0.025)
