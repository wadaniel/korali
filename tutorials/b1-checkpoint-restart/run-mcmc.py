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

k["Problem"]["Type"] = "Evaluation/Direct/Basic"
k["Problem"]["Objective Function"] = model

k["Solver"]["Type"]  = "Sampler/MCMC"
k["Solver"]["Burn In"] = 500
k["Solver"]["Console Frequency"] = 500
k["Solver"]["Save Frequency"] = 500

k["Solver"]["Termination Criteria"]["Max Samples"] = 2000
k["Solver"]["Termination Criteria"]["Generations Per Run"] = 1000

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Initial Mean"] = 0.0
k["Variables"][0]["Initial Standard Deviation"] = 1.0

k["Result Path"] = '_result_run-mcmc'

print("\n-------------------------------------------------------------")
print("Running first 1000 samples...")
print("-------------------------------------------------------------\n")

k.run()

print("\n-------------------------------------------------------------")
print("Running last 1000 samples...")
print("-------------------------------------------------------------\n")

k.run()
