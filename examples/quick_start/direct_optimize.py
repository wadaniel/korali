#!/usr/bin/env python3
import sys
sys.path.append('./model')
from directModel import *
import korali

k = korali.Engine()

k["Seed"] = 0xC0FFEE;
k["Verbosity"] = "Detailed"

k["Problem"]["Type"] = "Direct"
k["Problem"]["Variables"][0]["Name"] = "X"
k["Problem"]["Variables"][0]["Distribution"] = "Uniform"
k["Problem"]["Variables"][0]["Minimum"] = -10.0
k["Problem"]["Variables"][0]["Maximum"] = +10.0
k["Problem"]["Variables"][0]["Initial Mean"]   = 2.0
k["Problem"]["Variables"][0]["Initial StdDev"] = 5.0

k["Solver"]["Method"] = "CMA-ES"
k["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-8
k["Solver"]["Termination Criteria"]["Max Generations"] = 100
k["Solver"]["Lambda"] = 5
k["Solver"]["Objective"] = "Maximize"

k.run(evaluateModel)