#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
#from mpi4py import MPI
sys.path.append('./model')
from model import *

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = runEnvironment

# Defining the problem's variables.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"
e["Variables"][0]["Lower Bound"] = -1.0
e["Variables"][0]["Upper Bound"] = +1.0

e["Variables"][0]["Name"] = "A"
e["Variables"][0]["Type"] = "Action"
e["Variables"][0]["Lower Bound"] = -1.0
e["Variables"][0]["Upper Bound"] = +1.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "SARSA"
e["Solver"]["Environment Count"] = 10
e["Solver"]["Termination Criteria"]["Max Generations"] = 1

# Running Korali
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 3

#k["Conduit"]["Type"] = "Distributed"
#k["Conduit"]["Workers Per Team"] = 1
 
k.run(e)

