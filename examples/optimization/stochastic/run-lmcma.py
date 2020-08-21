#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
sys.path.append('./_model')
from model import *

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem.
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = negative_rosenbrock

dim = 3

# Defining the problem's variables.
for i in range(dim):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Lower Bound"] = -25.0
    e["Variables"][i]["Upper Bound"] = +25.0
    e["Variables"][i]["Initial Standard Deviation"] = 3.0

# Configuring LM-CMA parameters
e["Solver"]["Type"] = "Optimizer/LMCMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-15
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring results path
e["File Output"]["Path"] = '_korali_result_lmcma'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)
