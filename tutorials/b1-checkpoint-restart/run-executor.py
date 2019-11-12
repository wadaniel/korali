#!/usr/bin/env python3
import os
import sys
sys.path.append('./model/')
from model import *
import numpy as np

Ns        = 100
means     = [-5, 5, 10, 15, 20]
variances = [1, 1, 1, 1, 1]
fileName = '_executor_output/samplesOut.dat'
prepareFile(fileName)

import korali
k = korali.initialize()
e = korali.newExperiment()

e["Problem"]["Type"] = "Execution/Model"
e["Problem"]["Execution Model"] = lambda modelData: put_normal_rnds(modelData, Ns, fileName)

e["Variables"][0]["Name"] = "Mean"
e["Variables"][0]["Loaded Values"] = means
e["Variables"][1]["Name"] = "Variance"
e["Variables"][1]["Loaded Values"] = variances

e["Solver"]["Type"] = "Executor"
e["Solver"]["Executions Per Generation"] = 1
e["Solver"]["Termination Criteria"]["Generations Per Run"] = 2

e["Result Path"] = '_result_run-executor'
e["Verbosity"] = "Detailed"
e["Resume Previous"] = True

print("\n-------------------------------------------------------------")
print("Now running first 2 samples.")
print("-------------------------------------------------------------\n")

k.run(e)

print('------------------------------------------------------')
print("Now running another 2 samples.")
print('------------------------------------------------------')

k.run(e)

print('------------------------------------------------------')
print("Now running last sample.")
print('------------------------------------------------------')

k.run(e)
