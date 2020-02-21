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
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem.
e["Problem"]["Type"] = "Optimization/Stochastic"
e["Problem"]["Objective Function"] = model

# Defining the problem's variables.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0
e["Variables"][0]["Initial Mean"] = +0.0
e["Variables"][0]["Initial Standard Deviation"] = +2.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "LMCMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Running Korali
k.run(e)
