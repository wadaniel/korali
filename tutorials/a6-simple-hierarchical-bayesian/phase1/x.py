#!/usr/bin/env python3
import sys
import os
import shutil
sys.path.append('../setup/model')
from model import *
import korali

i = int(sys.argv[1])
dataPath    = "../setup/data/"
resultsPath = "../setup/results_phase_1/" + str(i).zfill(3)
if os.path.exists(resultsPath):  shutil.rmtree(resultsPath)
if not os.path.exists(resultsPath): os.makedirs(resultsPath)

# Running first Bayesian problem
k = korali.initialize()

k["Problem"]["Type"] = "Bayesian Inference"
k["Problem"]["Likelihood"]["Model"] = "Truncated Normal"
k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData(dataPath,i);

k["Problem"]["Variables"][0]["Name"] = "a"
k["Problem"]["Variables"][0]["Type"] = "Statistical"
k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = -10
k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] =  10
k["Problem"]["Variables"][0]["Lower Bound"] = -5.0
k["Problem"]["Variables"][0]["Upper Bound"] =  10.0

k["Problem"]["Variables"][1]["Name"] = "b"
k["Problem"]["Variables"][1]["Type"] = "Statistical"
k["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] = 0
k["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 20
k["Problem"]["Variables"][1]["Lower Bound"] =  10.0
k["Problem"]["Variables"][1]["Upper Bound"] =  25.0

k["Problem"]["Variables"][2]["Name"] = "mu"
k["Problem"]["Variables"][2]["Type"] = "Statistical"
k["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] =  5.0
k["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = 15.0
k["Problem"]["Variables"][2]["Lower Bound"] =  5.0
k["Problem"]["Variables"][2]["Upper Bound"] = 15.0

k["Problem"]["Variables"][3]["Name"] = "sigma"
k["Problem"]["Variables"][3]["Type"] = "Statistical"
k["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 10.0
k["Problem"]["Variables"][3]["Lower Bound"] = 0.0
k["Problem"]["Variables"][3]["Upper Bound"] = 15.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 20;
# k["Solver"]["Termination Criteria"]["Max Generations"] = 1

# k["Solver"]["Type"] = "CMAES"
# k["Solver"]["Population Size"] = 32
# k["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
# k["Solver"]["Termination Criteria"]["Max Generations"] = 100


k["General"]["Results Output"]["Path"] = resultsPath
k["General"]["Console Output"]["Verbosity"] = "Detailed"

k.run()
