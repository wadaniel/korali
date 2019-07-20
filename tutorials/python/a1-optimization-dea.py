#!/usr/bin/env python3

# In this example, we demonstrate how Korali finds values for the
# variables that maximize the objective function, given by a
# user-provided computational model.

# Importing computational model
import sys
sys.path.append('./model')
from directModel import *

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

# General Settings
k["General"]["Results Output"]["Path"] = "_a1_optimization_dea_result"
k["General"]["Random Seed"]  = 31415
k["General"]["Max Generations"] = 500

# Setting computational model
k.setModel(evaluateModel)

# Running Korali
k.run()
