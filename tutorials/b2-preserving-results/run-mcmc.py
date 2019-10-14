#!/usr/bin/env python3

## In this example, we demonstrate how to use Korali to sample
## a computational model, while saving important quantities of
## interest derived from the model

# Importing computational model
import sys
sys.path.append('model')
from model import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem and solver types.
k["Problem"]["Type"] = "Evaluation/Direct/Basic"
k["Problem"]["Objective Function"] = storeResults

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "Soil pH"
k["Variables"][0]["Initial Mean"] = 6.0
k["Variables"][0]["Initial Standard Deviation"] = 1.0

k["Variables"][1]["Name"] = "Average Temperature (C)"
k["Variables"][1]["Initial Mean"] = +30.0
k["Variables"][1]["Initial Standard Deviation"] = +5.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "Sampler/MCMC"
k["Solver"]["Burn In"] = 500
k["Solver"]["Termination Criteria"]["Max Chain Length"] = 1000

# Running Korali
k["Console Output"]["Frequency"] = 500
k["Results Output"]["Frequency"] = 500
k.run()

# Saving quantities of interest
saveResults()
