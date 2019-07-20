#!/usr/bin/env python3

# In this example, we demonstrate how Korali finds values for the
# variables that maximize the objective function, given by a
# user-provided computational model, subject to a set of
# constraints.

# Importing the computational model and constraints
import sys
sys.path.append('model')
from model import *
from constraints import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem type
k["Problem"]["Type"] = "Constrained Optimization"
k["Problem"]["Objective"] = "Maximize"

# Creating 7 variables and setting their CCMA-ES bounds
for i in range(7) :
  k["Variables"][i]["Name"] = "X" + str(i)
  k["Variables"][i]["Lower Bound"] = -10.0
  k["Variables"][i]["Upper Bound"] = +10.0

# Configuring the constrained optimizer CCMA-ES
k["Solver"]["Type"]  = "CCMAES"
k["Solver"]["Is Sigma Bounded"] = True
k["Solver"]["Sample Count"] = 32
k["Solver"]["Viability Sample Count"] = 4
k["Solver"]["Termination Criteria"]["Max Fitness"]["Enabled"] = True
k["Solver"]["Termination Criteria"]["Max Fitness"]["Value"] = -680.630057374402 - 1e-4

# General Settings
k["General"]["Max Generations"] = 500
k["General"]["Console Output"]["Frequency"] = 50
k["General"]["Results Output"]["Frequency"] = 50

# Setting model and constraints
k.setModel(model)
k.addConstraint(g1)
k.addConstraint(g2)
k.addConstraint(g3)
k.addConstraint(g4)

# Running Korali
k.run()
