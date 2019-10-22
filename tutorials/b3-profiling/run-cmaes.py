#!/usr/bin/env python3
import korali
import sys
sys.path.append("model")
from runModel import *

k = korali.initialize()

k["Problem"]["Type"] = "Evaluation/Direct/Basic";
k["Problem"]["Objective"] = "Maximize"
k["Problem"]["Objective Function"] = runModel

k["Solver"]["Type"] = "Optimizer/CMAES"
k["Solver"]["Population Size"] = 36
k["Solver"]["Termination Criteria"]["Max Generations"] = 10

k["Variables"][0]["Name"] = "X0"
k["Variables"][0]["Lower Bound"] = -32.0;
k["Variables"][0]["Upper Bound"] = +32.0;

k["Variables"][1]["Name"] = "X1"
k["Variables"][1]["Lower Bound"] = -32.0;
k["Variables"][1]["Upper Bound"] = +32.0;

k["Variables"][2]["Name"] = "X2"
k["Variables"][2]["Lower Bound"] = -32.0;
k["Variables"][2]["Upper Bound"] = +32.0;

k["Variables"][3]["Name"] = "X3"
k["Variables"][3]["Lower Bound"] = -32.0;
k["Variables"][3]["Upper Bound"] = +32.0;

#k["Conduit"]["Type"] = "External"
#k["Conduit"]["Concurrent Jobs"] = 4

k["Results Output"]["Profiling"] = "Enabled"
k["Random Seed"] = 0xC0FFEE

k.run()
