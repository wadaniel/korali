#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
for useDiagonalMetric in [False, True]:
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

    e["Solver"]["Termination Criteria"]["Max Samples"] = 10000

    # HMC specific parameters
    e["Solver"]["Version"] = 'Euclidean'
    e["Solver"]["Use Diagonal Metric"] = useDiagonalMetric
    e["Solver"]["Use Adaptive Step Size"] = True
    e["Solver"]["Use NUTS"] = True
    e["Solver"]["Target Acceptance Rate"] = 0.7

    # Running Korali
    e["Random Seed"] = 1337
    k.run(e)

    verifyMean(e["Solver"]["Sample Database"], [4.0], 0.25)
    verifyStd(e["Solver"]["Sample Database"], [4.0], 0.25)
