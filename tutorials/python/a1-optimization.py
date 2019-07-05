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

# Setting computational model
k.setModel(evaluateModel)

# Selecting problem and solver types.
k["Problem"] = "Direct Evaluation"
k["Solver"]  = "CMAES" 

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "X";
k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

# Configuring CMA-ES parameters
k["CMAES"]["Objective"] = "Maximize"
k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 500
k["CMAES"]["Sample Count"] = 5

# Setting output directory
k["Result Directory"] = "_a1_optimization_result"

# Running Korali
k.run()


