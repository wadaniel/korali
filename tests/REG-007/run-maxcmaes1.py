#!/usr/bin/env python3

import sys
sys.path.append('./model')
from model import *

import korali
k = korali.Engine()

e = korali.Experiment()

e["Problem"]["Type"] = "Evaluation/Direct/Basic"
e["Problem"]["Objective"] = "Maximize"
e["Problem"]["Objective Function"] = maxmodel1

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

k.run(e)

xopt = e["Solver"]["Internal"]["Best Ever Variables"]
fopt = e["Solver"]["Internal"]["Best Ever Value"]

x = e["Solver"]["Internal"]["Current Best Variables"]
f = e["Solver"]["Internal"]["Current Best Value"]

assertclose(2.0, xopt, 1e-3)
assertclose(2.0, x, 1e-3)
assertclose(-10.0, fopt, 1e-6)
assertclose(-10.0, f, 1e-4)
