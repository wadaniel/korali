#!/usr/bin/env python3
import sys
import os
import shutil
sys.path.append('../setup/model')
from model import *

i = int(sys.argv[1])
dataPath    = "../setup/data/"
resultsPath = "../setup/results_phase_1/" + str(i).zfill(3)
if os.path.exists(resultsPath):  shutil.rmtree(resultsPath)
if not os.path.exists(resultsPath): os.makedirs(resultsPath)

# Running first Bayesian problem
import korali
e = korali.newExperiment()

e["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Approximate"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData(dataPath,i);

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = 20.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = 10.0

e["Variables"][0]["Name"] = "mu"
e["Variables"][0]["Bayesian Type"] = "Statistical"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "sigma"
e["Variables"][1]["Bayesian Type"] = "Statistical"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 1000
e["Solver"]["Target Coefficient Of Variation"] = 0.6
e["Solver"]["Covariance Scaling"] = 0.02
e["Solver"]["Default Burn In"] = 0;

e["Result Path"] = resultsPath
e["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.initialize()
k.run(e)
