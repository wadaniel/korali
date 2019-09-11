#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior distribution
# in a bayesian problem where the likelihood is calculated by providing
# reference data points and their objective values.

# Importing the computational model
import sys
sys.path.append('./model')
from model import *
import korali

k = korali.initialize()

# Setting up the reference likelihood for the Bayesian Problem
k["Problem"]["Type"] = "Bayesian Inference (Reference Likelihood)"
k["Problem"]["Likelihood Model"] = "Additive Normal"
k["Problem"]["Reference Data"] = getReferenceData()
k["Problem"]["Computational Model"] = lambda sampleData: model(sampleData, getReferencePoints())

# Configuring TMCMC parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000

# Configuring the problem's variables and their prior distributions
k["Variables"][0]["Name"] = "a"
k["Variables"][0]["Bayesian Type"] = "Computational"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = -5.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0

k["Variables"][1]["Name"] = "b"
k["Variables"][1]["Bayesian Type"] = "Computational"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = +5.0

k["Variables"][2]["Name"] = "Sigma"
k["Variables"][2]["Bayesian Type"] = "Statistical"
k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = +5.0

# Running Korali
k.run()
