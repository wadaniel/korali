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

# Get Reference Points
x = getReferencePoints(dataPath,i);

# Running first Bayesian problem
k = korali.initialize()

k["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Reference"
k["Problem"]["Likelihood Model"] = "Additive Normal"
k["Problem"]["Reference Data"] = getReferenceData(dataPath,i);
k["Problem"]["Computational Model"] = lambda d: logistic( x, d);

k["Variables"][0]["Name"] = "C1"
k["Variables"][0]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

k["Variables"][1]["Name"] = "C2"
k["Variables"][1]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = 10.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 70.0

k["Variables"][2]["Name"] = "C3"
k["Variables"][2]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = 5.0

k["Variables"][3]["Name"] = "Sigma"
k["Variables"][3]["Bayesian Type"] = "Statistical"
k["Variables"][3]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][3]["Prior Distribution"]["Maximum"] = 30.0

k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 2000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 2;

k["Results Output"]["Path"] = resultsPath
k["Console Output"]["Verbosity"] = "Detailed"

k.run()
