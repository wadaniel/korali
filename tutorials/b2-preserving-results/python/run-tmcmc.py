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
k["Problem"]["Type"] = "Sampling"

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "Soil pH"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 3.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 9.0

k["Variables"][1]["Name"] = "Average Temperature (C)"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = +10.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = +50.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Target Coefficient of Variation"] = 0.2

# Setting Model
k.setModel(storeResults)

# Running Korali
k.run()

# Saving quantities of interest
saveResults()
