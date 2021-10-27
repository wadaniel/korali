#!/usr/bin/env python3

import sys
sys.path.append('_model')
from model import *
import korali

k = korali.Engine()
e = korali.Experiment()

e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = model

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 5
e["Solver"]["Termination Criteria"]["Max Generations"] = 50

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Console Output"]["Frequency"] = 25
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Specifying dry run to perform configuration checks but not actually running Korali
k["Dry Run"] = True
k.run(e)
