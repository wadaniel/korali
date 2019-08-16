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
k["Problem"]["Likelihood"]["Model"] = "Additive Normal"
k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData(dataPath,i);

k["Problem"]["Variables"][0]["Name"] = "C1"
k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

k["Problem"]["Variables"][1]["Name"] = "C2"
k["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] = 10.0
k["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 70.0

k["Problem"]["Variables"][2]["Name"] = "C3"
k["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = 5.0

k["Problem"]["Variables"][3]["Name"] = "Sigma"
k["Problem"]["Variables"][3]["Type"] = "Statistical"
k["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 30.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 2000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 3;

k["General"]["Results Output"]["Path"] = resultsPath
k["General"]["Console Output"]["Verbosity"] = "Detailed"

x = getReferencePoints(dataPath,i);
k.setReferenceModel( lambda d: logistic( x, d) )
k.run()
