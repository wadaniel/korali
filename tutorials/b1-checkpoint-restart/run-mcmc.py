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
e = korali.newExperiment()

e["Problem"]["Type"] = "Evaluation/Direct/Basic"
e["Problem"]["Objective Function"] = model

e["Solver"]["Type"]  = "Sampler/MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Termination Criteria"]["Max Samples"] = 2000
e["Solver"]["Termination Criteria"]["Generations Per Run"] = 1000

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

e["Result Path"] = '_result_run-mcmc'
e["Console Frequency"] = 500
e["Save Frequency"] = 500
e["Resume Previous"] = True

print("\n-------------------------------------------------------------")
print("Running first 1000 samples...")
print("-------------------------------------------------------------\n")

k.run(e)

print("\n-------------------------------------------------------------")
print("Running last 1000 samples...")
print("-------------------------------------------------------------\n")

k.run(e)
