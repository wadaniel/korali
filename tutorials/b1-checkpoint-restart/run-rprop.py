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
e = korali.newExperiment()

e["Problem"]["Type"] = "Evaluation/Direct/Gradient"
e["Problem"]["Objective"] = "Maximize"
e["Problem"]["Objective Function"] = model_with_gradient

e["Solver"]["Type"] = "Optimizer/Rprop"
e["Solver"]["Termination Criteria"]["Max Generations"] = 50
e["Solver"]["Termination Criteria"]["Generations Per Run"] = 25
e["Solver"]["Termination Criteria"]['Parameter Relative Tolerance'] = 1e-8;

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Value"] = -10.

e["Result Path"] = '_result_run-rprop'
e["Console Frequency"] = 10
e["Resume Previous"] = True

print("\n-------------------------------------------------------------")
print("Running first 25 generations...")
print("-------------------------------------------------------------\n")

k.run(e)

print("\n-------------------------------------------------------------")
print("Running last 25 generations...")
print("-------------------------------------------------------------\n")

k.run(e)

