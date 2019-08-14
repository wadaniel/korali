#!/usr/bin/env python3

# In this example, we demonstrate how a Korali experiment can
# be resumed from any point (generation). This is a useful feature
# for continuing jobs after an error, or to fragment big jobs into
# smaller ones that can better fit a supercomputer queue.

# First, we run a simple Korali experiment.

import sys
sys.path.append('model')
from model import *

import korali
k = korali.initialize()

resultDir = '_result_run-cmaes'

k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"

k["Problem"]["Variables"][0]["Name"] = "X"
k["Problem"]["Variables"][0]["Lower Bound"] = -10.0
k["Problem"]["Variables"][0]["Upper Bound"] = +10.0

k["Solver"]["Type"] = "CMAES"
k["Solver"]["Population Size"] = 5
k["Solver"]["Termination Criteria"]["Max Generations"] = 50

k["General"]["Console Output"]["Frequency"] = 10
k["General"]["Results Output"]["Path"] = resultDir

k.setDirectModel(model)

k.run()

print("\n-------------------------------------------------------------")
print("Now loading results from Gen 50 and running until Gen 100...")
print("-------------------------------------------------------------\n")

resultFile = korali.getLatestResult(resultDir)
k.loadState(resultFile)
k["Solver"]["Termination Criteria"]["Max Generations"] = 100
k.run()
