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

e["File Output"]["Enabled"] = False
e["Console Output"]["Frequency"] = 500

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = lexponential

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Initial Mean"] = 1.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 1000
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# HMC specific parameters
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Version"] = 'Euclidean'
e["Solver"]["Use Diagonal Metric"] = True
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Max Integration Steps"] = 1000
e["Solver"]["Use NUTS"] = False

# Running Korali
e["Random Seed"] = 1337
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [4.0], 0.2)
verifyStd(e["Solver"]["Sample Database"], [4.0], 0.1)
