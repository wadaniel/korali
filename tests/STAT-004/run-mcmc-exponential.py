#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
k = korali.Engine()
e = korali.Experiment()

e["Results"]["Path"] = "_result_run-mcmc"
e["Results"]["Frequency"] = 5000
e["Console"]["Frequency"] = 5000

# Selecting problem and solver types.
e["Problem"]["Type"] = "Evaluation/Direct/Basic"
e["Problem"]["Objective Function"] = lexponential

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the MCMC sampler parameters
e["Solver"]["Type"]  = "Sampler/MCMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Use Adaptive Sampling"] = False

e["Solver"]["Termination Criteria"]["Max Samples"] = 100000

# Running Korali
e["Random Seed"] = 1337
k.run(e)

verifyMean(e, [4.0], 0.05)
verifyStd(e, [4.0], 0.05)

