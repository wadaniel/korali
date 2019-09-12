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

k["Problem"]["Type"] = "Bayesian Inference (Approximate Likelihood)"
k["Problem"]["Likelihood Model"] = "Normal"
k["Problem"]["Reference Data"] = getReferenceData(dataPath,i);

k["Variables"][0]["Name"] = "mu"
k["Variables"][0]["Bayesian Type"] = "Statistical"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] =  0.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 20.0

k["Variables"][1]["Name"] = "sigma"
k["Variables"][1]["Bayesian Type"] = "Statistical"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 10.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 0;

k["Results Output"]["Path"] = resultsPath
k["Console Output"]["Verbosity"] = "Detailed"

k.run()
