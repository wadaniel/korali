#!/usr/bin/env python3

# Importing computational model
import sys
import korali
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Selecting problem and solver types.
e = korali.Experiment()
e["Problem"]["Type"] = "Optimization/Stochastic"
e["Problem"]["Objective Function"] = model

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000
e["Solver"]["Rejection Levels"] = 2
e["Solver"]["Use Adaptive Sampling"] = True

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 1.0
e["Variables"][0]["Initial Standard Deviation"] = 2.0

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["Console"]["Frequency"] = 500
e["Results"]["Frequency"] = 500
e["Results"]["Path"] = "_result_run-dram"

k = korali.Engine()
k.run(e)

checkMean(e, 0.0, 0.01)
checkStd(e, 1.0, 0.025)
