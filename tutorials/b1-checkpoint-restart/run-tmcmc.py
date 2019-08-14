#!/usr/bin/env python3

# In this example, we demonstrate how a Korali experiment can
# be resumed from any point (generation). This is a useful feature
# for continuing jobs after an error, or to fragment big jobs into
# smaller ones that can better fit a supercomputer queue.
#
# First, we run a simple Korali experiment.

import sys
sys.path.append('./model')
from model import *

import korali
k = korali.initialize()

resultDir = '_result_run-tmcmc'

k["Problem"]["Type"] = "Sampling"

k["Problem"]["Variables"][0]["Name"] = "X"
k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

k["Solver"]["Type"]  = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Termination Criteria"]["Max Generations"] = 1

k["General"]["Results Output"]["Path"] = resultDir
k["General"]["Random Seed"] = 0xC0FFEE

k.setDirectModel(model)
k.run()

print("\n-------------------------------------------------------------")
print("Now continuing from Generation 2 to end...")
print("-------------------------------------------------------------\n")

resultFile = korali.getLatestResult(resultDir)
k.loadState(resultFile)
k["Solver"]["Termination Criteria"]["Max Generations"] = 50
k.run()
