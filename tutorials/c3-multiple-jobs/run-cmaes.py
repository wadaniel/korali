#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
sys.path.append('./model')
from model import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Configuring Multiple Experiments.
experiments = []

for i in range(8):
 e = korali.newExperiment()
 e["Problem"]["Type"] = "Evaluation/Direct/Basic"
 e["Problem"]["Objective"] = "Maximize"
 e["Problem"]["Objective Function"] = model

 # Defining the problem's variables.
 e["Variables"][0]["Name"] = "X"
 e["Variables"][0]["Lower Bound"] = -10.0
 e["Variables"][0]["Upper Bound"] = +10.0

 # Configuring CMA-ES parameters
 e["Solver"]["Type"] = "Optimizer/CMAES"
 e["Solver"]["Population Size"] = 11
 e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
 e["Solver"]["Termination Criteria"]["Max Generations"] = 100
 
 # Setting distinct experiment paths
 e["Result Path"] = '_korali_multiple/exp' + str(i)
 
 # Adding Experiment to vector
 experiments.append(e)

k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = 8 

k.run(experiments)
