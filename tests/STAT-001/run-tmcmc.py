#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
from model import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Configuring problem
k["Problem"]["Type"] = "Sampling"

# Defining problem's variables and prior distribution for TMCMC
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"]["Type"] = "Gaussian"
k["Variables"][0]["Prior Distribution"]["Mean"] = 0.0
k["Variables"][0]["Prior Distribution"]["Standard Deviation"] = +5.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Covariance Scaling"] = 0.001

# Setting Model
k.setDirectModel(model)

# Running Korali
k["General"]["Random Seed"] = 0xC0FFEE
k["General"]["Results Output"]["Path"] = "_result_run-tmcmc"

k.run()
