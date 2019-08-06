#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the TMCMC method.

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
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Covariance Scaling"] = 0.001

# Setting Model
k.setDirectModel(model)

# Running Korali
k["General"]["Results Output"]["Path"] = "_result_run-tmcmc"
k.run()
