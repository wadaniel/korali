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
k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"

# Defining the problem's variables and their DEA bounds.
k["Variables"][0]["Name"] = "X";
k["Variables"][0]["Lower Bound"] = -10.0;
k["Variables"][0]["Upper Bound"] = +10.0;

# Configuring Solver
k["Solver"]["Type"] = "DEA"
k["Solver"]["Sample Count"] = 32
k["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"] = 1e-7

# Setting computational model
k.setModel(model)

# Running Korali
k.run()
