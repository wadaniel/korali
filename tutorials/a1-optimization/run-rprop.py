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

# Configuring Problem.
k["Problem"]["Type"] = "Evaluation/DirectWithGradient"
k["Problem"]["Objective"] = "Minimize"
k["Problem"]["Objective Function"] = model_with_gradient

# Defining the problem's variables.
for i in range(5):
  k["Variables"][i]["Name"] = "X" + str(i)
  k["Variables"][i]["Initial Value"] = -10.0+i

# Configuring CMA-ES parameters
k["Solver"]["Type"] = "Optimizer/Rprop"
k["Solver"]["Termination Criteria"]["Max Generations"] = 100
k["Solver"]["Termination Criteria"]["Parameter Relative Tolerance"] = 1e-8

# Running Korali
k.run()
