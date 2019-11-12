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
e = korali.newExperiment()

e["Result Path"] = "_result_run-tmcmc"

# Setting up custom likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Custom"
e["Problem"]["Likelihood Model"] = lgaussianCustom

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 20000


# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -15.0
e["Distributions"][0]["Maximum"] = +15.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Bayesian Type"] = "Computational"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

# Running Korali
e["Random Seed"] = 5467
k.run(e)

verifyMean(e, [-2.0], 0.05)
verifyStd(e, [3.0], 0.05)
