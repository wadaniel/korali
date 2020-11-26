#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
k = korali.Engine()
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = llaplace

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 1000
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# HMC specific parameters
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.05
e["Solver"]["Version"] = 'Euclidean'
e["Solver"]["Use Diagonal Metric"] = True
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Target Acceptance Rate"] = 0.8
e["Solver"]["Use NUTS"] = True

e["File Output"]["Enabled"] = False
e["Console Output"]["Frequency"] = 500

# Running Korali
e["Random Seed"] = 1227
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [4.0], 0.25)
verifyStd(e["Solver"]["Sample Database"], [math.sqrt(2)], 0.25)
