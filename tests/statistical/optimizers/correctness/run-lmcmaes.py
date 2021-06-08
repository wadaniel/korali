#!/usr/bin/env python3
import os
import sys
import korali

sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

#################################################
# LMCMAES problem definition & run
#################################################

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/LMCMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 200
e["Solver"]["Mu Type"] = "Linear"
  
e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 5e-2)

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/LMCMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 200
e["Solver"]["Mu Type"] = "Equal"
  
e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 5e-2)

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/LMCMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 200
e["Solver"]["Mu Type"] = "Logarithmic"
  
e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 5e-2)

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/LMCMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 200
e["Solver"]["Is Sigma Bounded"] = True
  
e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.22942553779431113, 5e-2)