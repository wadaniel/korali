#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
sys.path.append('./model')
from qoi import *

def evaluateModel(kdata):
 x = estimateHarvest(kdata.getVariable(0), kdata.getVariable(1))
 kdata.addResult(x)

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem and solver types.
k["Problem"] = "Bayesian"
k["Solver"]  = "TMCMC"
k["Bayesian"]["Likelihood"]["Type"] = "Direct"
k.setLikelihood(evaluateModel)

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "Soil pH"
#k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Gaussian"
#k["Variables"][0]["Bayesian"]["Prior Distribution"]["Mean"] = +6.5
#k["Variables"][0]["Bayesian"]["Prior Distribution"]["Sigma"] = +6.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = 3.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = 9.0

k["Variables"][1]["Name"] = "Average Temperature (C)"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = +10.0
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +50.0

# Configuring the TMCMC sampler parameters
k["TMCMC"]["Population Size"] = 5000

# Setting output directory
k["Result Directory"] = "_b3_quantities_of_interest_result"

# Running Korali
k.run()
