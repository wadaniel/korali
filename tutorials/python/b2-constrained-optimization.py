#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model, subject to a set of
## constraints.

# Importing the computational model and constraints
import sys
sys.path.append('./model')
from g09 import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Setting model and constraints
k.setModel( g09 )
k.addConstraint( g1 )
k.addConstraint( g2 )
k.addConstraint( g3 )
k.addConstraint( g4 )

# Selecting problem type.
k["Problem"] = "Direct Evaluation"

# Selecting the CCMA-ES solver.
k["Solver"]  = "CMA-ES"
k["CMA-ES"]["Objective"] = "Maximize"

# Creating 7 variables and
# Setting up the variables CCMA-ES bounds
nParams = 7
for i in range(nParams) :
  k["Variables"][i]["Name"] = "X" + str(i)
  k["Variables"][i]["CMA-ES"]["Lower Bound"] = -10.0
  k["Variables"][i]["CMA-ES"]["Upper Bound"] = +10.0

# Configuring the constrained optimizer CCMA-ES
k["CMA-ES"]["Sigma Bounded"] = True
k["CMA-ES"]["Sample Count"] = 8
k["CMA-ES"]["Constraint"]["Viability"]["Sample Count"] = 2
k["CMA-ES"]["Constraint"]["Adaption Size"] = 0.1
k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"] = True
k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 5000
k["CMA-ES"]["Termination Criteria"]["Max Fitness"]["Active"] = True
k["CMA-ES"]["Termination Criteria"]["Max Fitness"]["Value"] = -680.630057374402 - 1e-4

# Reducing Output
k["CMA-ES"]["Result Output Frequency"] = 100

# Setting output directory
k["Result Directory"] = "_b2_constrained_optimization"

# Running Korali
k.run()
