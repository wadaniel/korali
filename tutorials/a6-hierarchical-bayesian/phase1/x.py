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
k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian"
k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData(dataPath,i);

k["Variables"][0]["Name"] = "C1"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

k["Variables"][1]["Name"] = "C2"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = 10.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 70.0

k["Variables"][2]["Name"] = "C3"
k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = 5.0

k["Variables"][3]["Name"] = "Sigma"
k["Variables"][3]["Type"] = "Statistical"
k["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][3]["Prior Distribution"]["Maximum"] = 30.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.1
k["Solver"]["Termination Criteria"]["Max Generations"] = 100

k["General"]["Results Output"]["Path"] = resultsPath
k["General"]["Console Output"]["Verbosity"] = "Detailed"

x = getReferencePoints(dataPath,i);
k.setModel( lambda d: logistic( x, d) )
k.run()
