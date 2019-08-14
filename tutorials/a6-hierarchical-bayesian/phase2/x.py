#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"]  = "Hierarchical Bayesian (Psi)"
resultsPath = "../setup/results_phase_1/"
for i in range(10):
  kH["Problem"]["Sub Problems"][i] = resultsPath + str(i).zfill(3) + '/final.json'

# Add probability of theta given psi, one per subproblem variable.
kH["Problem"]["Conditional Priors"][0]["Type"] = "Uniform"
kH["Problem"]["Conditional Priors"][0]["Minimum"] = 280
kH["Problem"]["Conditional Priors"][0]["Maximum"] = 320

kH["Problem"]["Conditional Priors"][1]["Type"] = "Gaussian"
kH["Problem"]["Conditional Priors"][1]["Mean"] = "Psi 1"
kH["Problem"]["Conditional Priors"][1]["Standard Deviation"] = "Psi 2"

kH["Problem"]["Conditional Priors"][2]["Type"] = "LogNormal"
kH["Problem"]["Conditional Priors"][2]["Mu"]    = "Psi 3"
kH["Problem"]["Conditional Priors"][2]["Sigma"] = "Psi 4"

kH["Problem"]["Conditional Priors"][3]["Type"] = "Uniform"
kH["Problem"]["Conditional Priors"][3]["Minimum"] = "Psi 5"
kH["Problem"]["Conditional Priors"][3]["Maximum"] = "Psi 6"

kH["Problem"]["Variables"][0]["Name"] = "Psi 1"
kH["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = 10.0
kH["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = 70.0

kH["Problem"]["Variables"][1]["Name"] = "Psi 2"
kH["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] =  0.001
kH["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 30.0

kH["Problem"]["Variables"][2]["Name"] = "Psi 3"
kH["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = -1.0
kH["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = +1.0

kH["Problem"]["Variables"][3]["Name"] = "Psi 4"
kH["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
kH["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 10.0

kH["Problem"]["Variables"][4]["Name"] = "Psi 5"
kH["Problem"]["Variables"][4]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][4]["Prior Distribution"]["Minimum"] = 0.0
kH["Problem"]["Variables"][4]["Prior Distribution"]["Maximum"] = 15.0

kH["Problem"]["Variables"][5]["Name"] = "Psi 6"
kH["Problem"]["Variables"][5]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][5]["Prior Distribution"]["Minimum"] = 0.0
kH["Problem"]["Variables"][5]["Prior Distribution"]["Maximum"] = 15.0

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000
kH["Solver"]["Default Burn In"] = 3;
kH["Solver"]["Target Coefficient Of Variation"] = 0.6
kH["Solver"]["Covariance Scaling"] = 0.005

kH["General"]["Console Output"]["Verbosity"] = "Detailed"
kH["General"]["Results Output"]["Path"] = "../setup/results_phase_2/"

kH.run()
