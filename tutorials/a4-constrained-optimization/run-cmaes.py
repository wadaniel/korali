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
k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"
k["Problem"]["Objective Function"] = model
k["Problem"]["Constraints"][0] = g1
k["Problem"]["Constraints"][1] = g2
k["Problem"]["Constraints"][2] = g3
k["Problem"]["Constraints"][3] = g4

# Creating 7 variables and setting their CCMA-ES bounds
for i in range(7) :
  k["Variables"][i]["Name"] = "X" + str(i)
  k["Variables"][i]["Lower Bound"] = -10.0
  k["Variables"][i]["Upper Bound"] = +10.0

# Configuring the constrained optimizer CCMA-ES
k["Solver"]["Type"] = "CMAES"
k["Solver"]["Is Sigma Bounded"] = True
k["Solver"]["Population Size"] = 32
k["Solver"]["Viability Population Size"] = 4
k["Solver"]["Termination Criteria"]["Max Value"] = -680.630057374402 - 1e-4
k["Solver"]["Termination Criteria"]["Max Generations"] = 500

# General Settings
k["Console Output"]["Frequency"] = 50
k["Results Output"]["Frequency"] = 50

# Running Korali
k.run()
