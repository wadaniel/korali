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
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Initial Value"] = -10.0

# Configuring CMA-ES parameters
k["Solver"]["Type"] = "Optimizer/Rprop"

# Running Korali
k.run()
