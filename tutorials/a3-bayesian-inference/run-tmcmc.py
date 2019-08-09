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
k["Problem"]["Type"] = "Bayesian Inference"
k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian"
k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData()

# Configuring the problem's variables and their prior distributions
k["Problem"]["Variables"][0]["Name"] = "a"
k["Problem"]["Variables"][0]["Type"] = "Computational"
k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = -5.0
k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0

k["Problem"]["Variables"][1]["Name"] = "b"
k["Problem"]["Variables"][1]["Type"] = "Computational"
k["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0
k["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = +5.0

k["Problem"]["Variables"][2]["Name"] = "Sigma"
k["Problem"]["Variables"][2]["Type"] = "Statistical"
k["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = +5.0

# Configuring TMCMC parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000

# Setting the model
k.setReferenceModel(lambda modelData: model(modelData, getReferencePoints()))

# Running Korali
k.run()
