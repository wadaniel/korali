#!/usr/bin/env python3

# In this example, we demonstrate how a Korali experiment can
# be resumed from any point (generation). This is a useful feature
# for continuing jobs after an error, or to fragment big jobs into
# smaller ones that can better fit a supercomputer queue.

# First, we run a simple Korali experiment.

import sys
sys.path.append('model')
from model import *

import korali
k = korali.initialize()

k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Lower Bound"] = -10.0
k["Variables"][0]["Upper Bound"] = +10.0

k["Solver"]["Type"] = "CMAES"
k["Solver"]["Sample Count"] = 5

k["General"]["Max Generations"] = 500
k["General"]["Console Output"]["Frequency"] = 10
k["General"]["Results Output"]["Path"] = "_result_run-cmaes"

k.setModel(model)

k.run()

print("\n\nRestarting Now...\n\n")

# Now we loadState() to resume the same experiment from generation 10
k.loadState("_result_run-cmaes/s00010.json")

k.run()
