#!/usr/bin/env python3

# In this example, we demonstrate how Korali finds values for the
# variables that maximize the objective function, given by a
# user-provided computational model.

# Importing computational model
import sys
sys.path.append('./model')
from model import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Configuring Problem
k["Problem"]["Type"] = "Evaluation/Direct"
k["Problem"]["Objective"] = "Maximize"
k["Problem"]["Objective Function"] = model

# Configuring Solver
k["Solver"]["Type"] = "Optimizer/DEA"
k["Solver"]["Population Size"] = 32
k["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
k["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Defining the problem's variables and their bounds.
k["Variables"][0]["Name"] = "X";
k["Variables"][0]["Lower Bound"] = -10.0;
k["Variables"][0]["Upper Bound"] = +10.0;

# Running Korali
k.run()
