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

k["Distributions"][0]["Name"] = "Uniform 0"
k["Distributions"][0]["Type"] = "Univariate/Uniform"
k["Distributions"][0]["Minimum"] = 10.0
k["Distributions"][0]["Maximum"] = 70.0

k["Distributions"][1]["Name"] = "Uniform 1"
k["Distributions"][1]["Type"] = "Univariate/Uniform"
k["Distributions"][1]["Minimum"] = 0.001
k["Distributions"][1]["Maximum"] = 30.0

k["Distributions"][2]["Name"] = "Uniform 2"
k["Distributions"][2]["Type"] = "Univariate/Uniform"
k["Distributions"][2]["Minimum"] = -1.0
k["Distributions"][2]["Maximum"] = +1.0

k["Distributions"][3]["Name"] = "Uniform 3"
k["Distributions"][3]["Type"] = "Univariate/Uniform"
k["Distributions"][3]["Minimum"] = 0.0
k["Distributions"][3]["Maximum"] = 10.0

k["Distributions"][4]["Name"] = "Uniform 4"
k["Distributions"][4]["Type"] = "Univariate/Uniform"
k["Distributions"][4]["Minimum"] = 0.0
k["Distributions"][4]["Maximum"] = 15.0

k["Distributions"][5]["Name"] = "Uniform 5"
k["Distributions"][5]["Type"] = "Univariate/Uniform"
k["Distributions"][5]["Minimum"] = 0.0
k["Distributions"][5]["Maximum"] = 15.0

k["Variables"][0]["Name"] = "Psi 1"
k["Variables"][0]["Prior Distribution"] = "Uniform 0"

k["Variables"][1]["Name"] = "Psi 2"
k["Variables"][1]["Prior Distribution"] = "Uniform 1"

k["Variables"][2]["Name"] = "Psi 3"
k["Variables"][2]["Prior Distribution"] = "Uniform 2"

k["Variables"][3]["Name"] = "Psi 4"
k["Variables"][3]["Prior Distribution"] = "Uniform 3"

k["Variables"][4]["Name"] = "Psi 5"
k["Variables"][4]["Prior Distribution"] = "Uniform 4"

k["Variables"][5]["Name"] = "Psi 6"
k["Variables"][5]["Prior Distribution"] = "Uniform 5"

k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Default Burn In"] = 3;
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.01

k["Random Seed"] = 0xC0FFEE
k["Console Output"]["Verbosity"] = "Detailed"
k["Results Output"]["Path"] = "../setup/results_phase_2/"

k.run()
