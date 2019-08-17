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
kH["Problem"]["Conditional Priors"][0]["Type"] = "Normal"
kH["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1"
kH["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2"

kH["Problem"]["Conditional Priors"][1]["Type"] = "LogNormal"
kH["Problem"]["Conditional Priors"][1]["Mu"]    = "Psi 3"
kH["Problem"]["Conditional Priors"][1]["Sigma"] = "Psi 4"


kH["Problem"]["Variables"][0]["Name"] = "Psi 1"
kH["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] =  0.0
kH["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = 20.0

kH["Problem"]["Variables"][1]["Name"] = "Psi 2"
kH["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] =  0.0
kH["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 20.0

kH["Problem"]["Variables"][2]["Name"] = "Psi 3"
kH["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = -2.0
kH["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = +2.0

kH["Problem"]["Variables"][3]["Name"] = "Psi 4"
kH["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
kH["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 5.0


kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 2000
kH["Solver"]["Default Burn In"] = 1;
kH["Solver"]["Target Coefficient Of Variation"] = 0.6
kH["Solver"]["Covariance Scaling"] = 0.01

kH["General"]["Console Output"]["Verbosity"] = "Detailed"
kH["General"]["Results Output"]["Path"] = "../setup/results_phase_2/"

kH.run()
