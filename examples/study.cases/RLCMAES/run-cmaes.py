#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
import math
sys.path.append('./_environment')
from env import *

dim = 2
populationSize = 16

objective = ObjectiveFactory(populationSize)
objective.reset()
print(objective.function([0., 0.]))

fun = lambda s: objective.evaluateNegative(s)

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = fun

# Defining the problem's variables.
for i in range(dim):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Initial Value"] = 0.
    e["Variables"][i]["Initial Standard Deviation"] = 1.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = populationSize
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)


print("Objective: {}".format(objective.name))
#print("Initial Ef {} -- Terminal Ef {}".format(objective.initialEf, objective.curEf))
#print("Initial Best F {} -- Terminal Best F {} -- Best Ever F {}".format(objective.initialBestF, objective.curBestF, objective.bestEver))
 
