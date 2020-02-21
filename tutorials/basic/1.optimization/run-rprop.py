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
e["Problem"]["Type"] = "Optimization/Gradient"
e["Problem"]["Objective Function"] = model_with_gradient

# Defining the problem's variables.
for i in range(5):
  e["Variables"][i]["Name"] = "X" + str(i)
  e["Variables"][i]["Initial Value"] = -10.0+i

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/Rprop"
e["Solver"]["Termination Criteria"]["Max Generations"] = 200
e["Solver"]["Termination Criteria"]["Parameter Relative Tolerance"] = 1e-8

# Running Experiment
k.run(e)
