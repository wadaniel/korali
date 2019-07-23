#!/usr/bin/env python3
import sys
import os
sys.path.append('setup/model')
from model import *
import korali

# Running first Bayesian problem
k = korali.initialize()

k["Problem"]["Type"] = "Bayesian Inference"
k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian"

k["Variables"][0]["Name"] = "C1"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

k["Variables"][1]["Name"] = "C2"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = 10.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 70.0

k["Variables"][2]["Name"] = "C3"
k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = 5.0

k["Variables"][3]["Name"] = "Sigma"
k["Variables"][3]["Type"] = "Statistical"
k["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][3]["Prior Distribution"]["Maximum"] = 10.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000

k["General"]["Console Output"]["Verbosity"] = "Detailed"
k["General"]["Max Generations"] = 100

for i in range(3):
  dataPath = "setup/data/individual"

  resultsPath = "results/individual/" + str(i+1).zfill(3)
  if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)
  k["General"]["Results Output"]["Path"] = resultsPath
  y = getReferenceData(dataPath,i+1);
  k["Problem"]["Likelihood"]["Reference Data"] = y
  k.setModel( lambda d: logistic( getReferencePoints(dataPath,i+1), d) )
  k.dry()
