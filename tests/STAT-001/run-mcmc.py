#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem and solver types.
k["Problem"]["Type"] = "Evaluation/Direct/Basic"
k["Problem"]["Objective Function"] = model

# Configuring the MCMC sampler parameters
k["Solver"]["Type"] = "Sampler/MCMC"
k["Solver"]["Burn In"] = 500
k["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# Defining problem's variables and their MCMC settings
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Initial Mean"] = 1.0
k["Variables"][0]["Initial Standard Deviation"] = 1.0

# Running Korali
k["Random Seed"] = 0xC0FFEE
k["Results Output"]["Path"] = "_result_run-mcmc"
k.run()

# Testing Results
checkMean(k, 0.0, 0.01)
checkStd(k, 1.0, 0.025)
