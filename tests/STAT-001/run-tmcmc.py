#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
from model import *

# Starting Korali's Engine
import korali
k = korali.initialize()
k["Random Seed"] = 0xC0FFEE
k["Results Output"]["Path"] = "_result_run-tmcmc"

# Configuring problem
k["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Custom"
k["Problem"]["Likelihood Model"] = model

# Defining problem's variables and prior distribution for TMCMC
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = -100000.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = +100000.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Covariance Scaling"] = 0.04

# Running Korali
k.run()
