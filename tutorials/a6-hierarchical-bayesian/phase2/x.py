#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali


# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"]  = "Hierarchical Bayesian"
kH["Problem"]["Model"] = "Sample Psi"
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
kH["Problem"]["Conditional Priors"][3]["Minimum"] = 0.1
kH["Problem"]["Conditional Priors"][3]["Maximum"] = 15.

kH["Problem"]["Variables"][0]["Name"] = "Psi 1"
kH["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = 20.0
kH["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = 60.0

kH["Problem"]["Variables"][1]["Name"] = "Psi 2"
kH["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] =  0.001
kH["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 20.0

kH["Problem"]["Variables"][2]["Name"] = "Psi 3"
kH["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = -1.0
kH["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = +1.0

kH["Problem"]["Variables"][3]["Name"] = "Psi 4"
kH["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
kH["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0.0
kH["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 10.0

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000
kH["Solver"]["Termination Criteria"]["Max Generations"] = 10
# kH["Solver"]["Burn In Steps"] = [ 3 ];
kH["Solver"]["Target Coefficient Of Variation"] = 0.6

kH["General"]["Console Output"]["Verbosity"] = "Detailed"

kH.run()
