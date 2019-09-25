#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Creating hierarchical Bayesian problem from previous two problems
k = korali.initialize()

k["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/Psi"
resultsPath = "../setup/results_phase_1/"
for i in range(5):
  k["Problem"]["Sub Problems"][i] = resultsPath + str(i).zfill(3) + '/final.json'

# Add probability of theta given psi, one per subproblem variable.
k["Problem"]["Conditional Priors"][0]["Type"] = "Univariate/Normal"
k["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1"
k["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2"

k["Problem"]["Conditional Priors"][1]["Type"] = "Univariate/LogNormal"
k["Problem"]["Conditional Priors"][1]["Mu"]    = "Psi 3"
k["Problem"]["Conditional Priors"][1]["Sigma"] = "Psi 4"

k["Variables"][0]["Name"] = "Psi 1"
k["Variables"][0]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] =  0.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 20.0

k["Variables"][1]["Name"] = "Psi 2"
k["Variables"][1]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] =  0.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 20.0

k["Variables"][2]["Name"] = "Psi 3"
k["Variables"][2]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = -2.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = +2.0

k["Variables"][3]["Name"] = "Psi 4"
k["Variables"][3]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][3]["Prior Distribution"]["Maximum"] = 5.0

k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 2000
k["Solver"]["Default Burn In"] = 1;
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.1

k["Console Output"]["Verbosity"] = "Detailed"
k["Results Output"]["Path"] = "../setup/results_phase_2/"

k.run()
