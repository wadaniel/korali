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
k["Solver"]  = "DE" 

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "X";
k["Variables"][0]["DE"]["Lower Bound"] = -10.0;
k["Variables"][0]["DE"]["Upper Bound"] = +10.0;

# Configuring CMA-ES parameters
#k["DE"]["Objective"] = "Maximize"
k["DE"]["Termination Criteria"]["Max Generations"]["Value"] = 500
k["DE"]["Sample Count"] = 32

# Setting output directory
k["Result Directory"] = "_a1_optimization_result"

# Running Korali
k.run()


