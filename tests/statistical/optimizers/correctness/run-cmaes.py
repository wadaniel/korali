#!/usr/bin/env python3
import os
import sys
import korali

sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

#################################################
# CMAES problem definition & run
#################################################

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 1e-4)

### Running With Diagonal Covariance Matrix

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Diagonal Covariance"] = True

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 1e-4)

### Running With Mirrored Sampling

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Mirrored Sampling"] = True

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 1e-4)