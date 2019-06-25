#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
sys.path.append('./model')
from directModel import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem and solver types.
k["Problem"] = "Direct Evaluation"
k["Solver"]  = "CMA-ES" 

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "X";
k["Variables"][0]["CMA-ES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CMA-ES"]["Upper Bound"] = +10.0;

# Configuring CMA-ES parameters
k["CMA-ES"]["Objective"] = "Maximize"
k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 500
k["CMA-ES"]["Sample Count"] = 5

# Setting computational model
k.setModel(evaluateModel)

# Running Korali
k.run()
