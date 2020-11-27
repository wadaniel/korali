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
  e["File Output"]["Enabled"] = False
  e["Console Output"]["Frequency"] = 500

  # Defining problem's variables and their HMC settings
  for i in range(5):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Initial Mean"] = -1.0
    e["Variables"][i]["Initial Standard Deviation"] = 1.0

  # Configuring the HMC sampler parameters
  e["Solver"]["Type"] = "Sampler/HMC"
  e["Solver"]["Burn In"] = 1000
  e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

  # HMC specific parameters
  e["Solver"]["Num Integration Steps"] = 20
  e["Solver"]["Step Size"] = 0.1
  e["Solver"]["Version"] = 'Euclidean'
  e["Solver"]["Use Diagonal Metric"] = useDiagonalMetric
  e["Solver"]["Use Adaptive Step Size"] = True
  e["Solver"]["Max Integration Steps"] = 1000
  e["Solver"]["Use NUTS"] = False

  # Running Korali
  e["Random Seed"] = 1337
  k.run(e)

  verifyMean(e["Solver"]["Sample Database"], [0.0, 0.0, 0.0, 0.0, 0.0], 0.1)
  verifyStd(e["Solver"]["Sample Database"], [1.0, 1.0, 1.0, 1.0, 1.0], 0.1)
