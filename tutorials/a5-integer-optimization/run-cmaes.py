#!/usr/bin/env python3

# In this example, we demonstrate how Korali finds values for the
# variables that maximize the objective function, given by a
# user-provided computational model, subject to a set of
# constraints.

# Importing the computational model and constraints
import sys
sys.path.append('model')
from model import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem type
k["Problem"]["Type"] = "Evaluation/Direct/Basic"
k["Problem"]["Objective"] = "Maximize"
k["Problem"]["Objective Function"] = model

# Creating 10 variables and setting their CMA-ES bounds
for i in range(10) :
  k["Variables"][i]["Name"] = "X" + str(i)
  k["Variables"][i]["Initial Mean"] = 1.0
  k["Variables"][i]["Lower Bound"]  = -19.0
  k["Variables"][i]["Upper Bound"]  = +21.0

# We set some of them as discrete.
k["Variables"][0]["Granularity"] = 1.0
k["Variables"][1]["Granularity"] = 1.0
k["Variables"][3]["Granularity"] = 1.0
k["Variables"][6]["Granularity"] = 1.0

# Configuring CMA-ES parameters
k["Solver"]["Type"] = "Optimizer/CMAES"
k["Solver"]["Population Size"] = 8
k["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-9
k["Solver"]["Termination Criteria"]["Max Generations"] = 5000

k["Results Output"]["Frequency"] = 50
k["Console Output"]["Frequency"] = 50

k.run()
