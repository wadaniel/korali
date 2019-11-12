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
k["Random Seed"] = 0xC0FFEE
k["Results Output"]["Path"] = "_result_run-tmcmc2"
#k["Console Output"]["Verbosity"] = "Detailed"

# Configuring problem
k["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Custom"
k["Problem"]["Likelihood Model"] = model

k["Distributions"][0]["Name"] = "Uniform 0"
k["Distributions"][0]["Type"] = "Univariate/Uniform"
k["Distributions"][0]["Minimum"] = -20.0
k["Distributions"][0]["Maximum"] = +20.0
 
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"] = "Uniform 0"

# Defining problem's variables and prior distribution for TMCMC
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"] = "Uniform 0"

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Covariance Scaling"] = 0.04
k["Solver"]["Default Burn In"] = 5
k["Solver"]["Target Coefficient Of Variation"] = 0.6

# Running Korali
k.run()

checkMean(k, 0.0, 0.02)
checkStd(k, 1.0, 0.025)
