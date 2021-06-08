#!/usr/bin/env python3
import os
import sys
import korali

sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Creating value list
values = np.linspace(-10, 10, 1000).tolist()

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Values"] = values

e["Solver"]["Type"] = "Optimizer/GridSearch"
e["Solver"]["Termination Criteria"]["Max Generations"] = 2

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 1e-4)

