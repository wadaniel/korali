#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./_model')
from model import *
import numpy as np

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Creating value list
values = np.linspace(-1, 1, 10).tolist()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = modelGrid

# Defining the problem's variables.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Values"] = values
e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Values"] = values
e["Variables"][2]["Name"] = "Z"
e["Variables"][2]["Values"] = values

# Chosing Grid Search solver
e["Solver"]["Type"] = "Optimizer/GridSearch"

# Running Korali
k.run(e)
