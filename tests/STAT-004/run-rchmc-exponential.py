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

    e["File Output"]["Frequency"] = 0
    e["Console Output"]["Frequency"] = 5000

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
    e["Solver"]["Termination Criteria"]["Max Samples"] = 100000

    # HMC specific parameters
    e["Solver"]["Num Integration Steps"] = 5
    e["Solver"]["Step Size"] = 0.01
    e["Solver"]["Version"] = 'Riemannian Const'
    e["Solver"]["Use Diagonal Metric"] = useDiagonalMetric
    e["Solver"]["Use Adaptive Step Size"] = False
    e["Solver"]["Target Integration Time"] = 0.70
    e["Solver"]["Step Size Jitter"] = 0.1
    e["Solver"]["Desired Average Acceptance Rate"] = 0.65
    e["Solver"]["Use NUTS"] = False

    e["Console Output"]["Verbosity"] = "Detailed"
    # Running Korali
    e["Random Seed"] = 1337
    k.run(e)

    verifyMean(e["Solver"]["Sample Database"], [4.0], 0.05)
    verifyStd(e["Solver"]["Sample Database"], [4.0], 0.05)
