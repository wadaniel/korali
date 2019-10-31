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
k["Results Output"]["Path"] = "_result_run-mcmc"

# Selecting problem and solver types.
k["Problem"]["Type"] = "Evaluation/Direct/Basic"
k["Problem"]["Objective Function"] = lgaussian

# Defining problem's variables and their MCMC settings
k["Variables"][0]["Name"] = "X0"
k["Variables"][0]["Initial Mean"] = 0.0
k["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the MCMC sampler parameters
k["Solver"]["Type"]  = "Sampler/MCMC"
k["Solver"]["Burn In"] = 500
k["Solver"]["Use Adaptive Sampling"] = False
k["Solver"]["Termination Criteria"]["Max Samples"] = 10000

# Running Korali
k.run()

#verifyMean(k, [-2.0])
#verifyMean(k, [3.0])
