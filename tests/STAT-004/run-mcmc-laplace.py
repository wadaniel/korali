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

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = llaplace

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0


# Configuring the MCMC sampler parameters
e["Solver"]["Type"]  = "MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Use Adaptive Sampling"] = True
e["Solver"]["Termination Criteria"]["Max Samples"] = 100000

e["Console"]["Frequency"] = 5000
e["Results"]["Frequency"] = 5000
e["Results"]["Path"] = "_result_run-mcmc"

# Running Korali
e["Random Seed"] = 1227
k.run(e)

verifyMean(e, [4.0], 0.05)
verifyStd(e, [math.sqrt(2)], 0.05)

