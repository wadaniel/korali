#!/usr/bin/env python3

# In this example, we demonstrate how a Korali experiment can
# be resumed from any point (generation). This is a useful feature
# for continuing jobs after an error, or to fragment big jobs into
# smaller ones that can better fit a supercomputer queue.
#
# First, we run a simple Korali experiment.

import sys
sys.path.append('./model')
from model import *

import korali
k = korali.initialize()

k["Problem"]["Type"] = "Sampling"

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Initial Mean"] = 0.0
k["Variables"][0]["Initial Standard Deviation"] = 1.0

k["Solver"]["Type"]  = "MCMC"
k["Solver"]["Burn In"] = 500
k["Solver"]["Termination Criteria"]["Max Chain Length"] = 5000

k["General"]["Console Output"]["Frequency"] = 1000
k["General"]["Results Output"]["Frequency"] = 1000
k["General"]["Results Output"]["Path"] = "_result_run-mcmc"

k.setDirectModel(model)
k.run()

print("\n\nRestarting now:\n\n");

# Now we loadState() to resume the same experiment from generation 5.
k.loadState("_result_run-mcmc/s01000.json")

k.run()
