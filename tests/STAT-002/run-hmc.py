#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
e = korali.Experiment()
e["File Output"]["Path"] = "_result_run-HMC"

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = model
e["Console Output"]["Frequency"] = 500
e["File Output"]["Frequency"] = 500

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Termination Criteria"]["Max Samples"] = 10000
e["Solver"]["Adaptive Time Stepping"] = 0
e["Solver"]["Use Euclidean Metric"] = 0
e["Solver"]["Use NUTS"] = 0
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 20.0
e["Solver"]["Desired Average Acceptance Rate"] = 0.98

e["Console Output"]["Verbosity"] = "Detailed"

# Running Korali
k = korali.Engine()
k.run(e)

compareMean(e)
compareStd(e)
