#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

lg5 = lambda x: lgaussianxd(x, 5)

# Starting Korali's Engine
import korali
for useDiagonalMetric in [False, True]:
  k = korali.Engine()
  e = korali.Experiment()

  # Selecting problem and solver types.
  e["Problem"]["Type"] = "Sampling"
  e["Problem"]["Probability Function"] = lg5
  e["Console Output"]["Frequency"] = 5000
  e["File Output"]["Frequency"] = 0

  # Defining problem's variables and their HMC settings
  for i in range(5):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Initial Mean"] = -1.0
    e["Variables"][i]["Initial Standard Deviation"] = 1.0

  # Configuring the HMC sampler parameters
  e["Solver"]["Type"] = "Sampler/HMC"
  e["Solver"]["Burn In"] = 500
  e["Solver"]["Termination Criteria"]["Max Samples"] = 50000

  # HMC specific parameters
  e["Solver"]["Num Integration Steps"] = 20
  e["Solver"]["Step Size"] = 0.05
  e["Solver"]["Version"] = 'Euclidean'
  e["Solver"]["Use Diagonal Metric"] = useDiagonalMetric
  e["Solver"]["Use Adaptive Step Size"] = True
  e["Solver"]["Target Integration Time"] = 0.5
  e["Solver"]["Target Acceptance Rate"] = 0.80
  e["Solver"]["Use NUTS"] = False

  # Running Korali
  e["Random Seed"] = 1337
  k.run(e)

  verifyMean(e["Solver"]["Sample Database"], [0.0, 0.0, 0.0, 0.0, 0.0], 0.05)
  verifyStd(e["Solver"]["Sample Database"], [1.0, 1.0, 1.0, 1.0, 1.0], 0.05)
