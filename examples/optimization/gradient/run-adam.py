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

# Defining the problem's variables.
i = 0
e["Variables"][i]["Name"] = "X" + str(i)
e["Variables"][i]["Initial Value"] = -5

i = 1
e["Variables"][i]["Name"] = "X" + str(i)
e["Variables"][i]["Initial Value"] = 10.0


# Configuring Adam parameters
e["Solver"]["Type"] = "Optimizer/Adam"
e["Solver"]["Eta"] = 0.1
e["Solver"]["Termination Criteria"]["Max Generations"] = 5000

# Configuring results path
e["File Output"]["Path"] = '_korali_result_adam'

# Running Experiment
k.run(e)
