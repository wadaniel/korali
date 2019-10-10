#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Creating hierarchical Bayesian problem from previous two problems
k = korali.initialize()

k["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/Psi"
k["Problem"]["Sub Problems"] = ['../setup/results_phase_1/' + str(i).zfill(3) + '/final.json' for i in range(5)]

# Add probability of theta given psi, one per subproblem variable.
k["Problem"]["Conditional Priors"][0]["Type"] = "Univariate/Uniform"
k["Problem"]["Conditional Priors"][0]["Minimum"] = 280
k["Problem"]["Conditional Priors"][0]["Maximum"] = 320

k["Problem"]["Conditional Priors"][1]["Type"] = "Univariate/Normal"
k["Problem"]["Conditional Priors"][1]["Mean"] = "Psi 1"
k["Problem"]["Conditional Priors"][1]["Standard Deviation"] = "Psi 2"

k["Problem"]["Conditional Priors"][2]["Type"] = "Univariate/LogNormal"
k["Problem"]["Conditional Priors"][2]["Mu"]    = "Psi 3"
k["Problem"]["Conditional Priors"][2]["Sigma"] = "Psi 4"

k["Problem"]["Conditional Priors"][3]["Type"] = "Univariate/Uniform"
k["Problem"]["Conditional Priors"][3]["Minimum"] = "Psi 5"
k["Problem"]["Conditional Priors"][3]["Maximum"] = "Psi 6"

k["Variables"][0]["Name"] = "Psi 1"
k["Variables"][0]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 10.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 70.0

k["Variables"][1]["Name"] = "Psi 2"
k["Variables"][1]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] =  0.001
k["Variables"][1]["Prior Distribution"]["Maximum"] = 30.0

k["Variables"][2]["Name"] = "Psi 3"
k["Variables"][2]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = -1.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = +1.0

k["Variables"][3]["Name"] = "Psi 4"
k["Variables"][3]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][3]["Prior Distribution"]["Maximum"] = 10.0

k["Variables"][4]["Name"] = "Psi 5"
k["Variables"][4]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][4]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][4]["Prior Distribution"]["Maximum"] = 15.0

k["Variables"][5]["Name"] = "Psi 6"
k["Variables"][5]["Prior Distribution"]["Type"] = "Univariate/Uniform"
k["Variables"][5]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][5]["Prior Distribution"]["Maximum"] = 15.0

k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Default Burn In"] = 3;
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.01

k["Random Seed"] = 0xC0FFEE
k["Console Output"]["Verbosity"] = "Detailed"
k["Results Output"]["Path"] = "../setup/results_phase_2/"

k.run()
