#!/usr/bin/env python3

# Importing computational model
import sys
import korali
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Selecting problem and solver types.
e = korali.newExperiment()
e["Problem"]["Type"] = "Evaluation/Direct/Basic"
e["Problem"]["Objective Function"] = model

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 2.0

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Rejection Levels"] = 2
e["Solver"]["Termination Criteria"]["Max Samples"] = 10000

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["Console Frequency"] = 500
e["Save Frequency"] = 500
e["Result Path"] = "_result_run-dram"

k = korali.initialize()
k.run(e)

checkMean(e, 0.0, 0.05)
checkStd(e, 1.0, 0.05)
