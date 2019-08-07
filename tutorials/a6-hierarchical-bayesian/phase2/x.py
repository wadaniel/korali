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
kH["Problem"]["Sub Problems"][0] = resultsPath + '000/final.json'
kH["Problem"]["Sub Problems"][1] = resultsPath + '001/final.json'
kH["Problem"]["Sub Problems"][2] = resultsPath + '002/final.json'
kH["Problem"]["Sub Problems"][3] = resultsPath + '003/final.json'
kH["Problem"]["Sub Problems"][4] = resultsPath + '004/final.json'

# Add probability of theta given psi, one per subproblem variable.
kH["Problem"]["Conditional Priors"][0]["Type"] = "Gaussian"
kH["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1"
kH["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2"

kH["Problem"]["Conditional Priors"][1]["Type"] = "Gaussian"
kH["Problem"]["Conditional Priors"][1]["Mean"] = "Psi 3"
kH["Problem"]["Conditional Priors"][1]["Standard Deviation"] = "Psi 4"

kH["Problem"]["Conditional Priors"][2]["Type"] = "LogNormal"
kH["Problem"]["Conditional Priors"][2]["Mu"]    = "Psi 5"
kH["Problem"]["Conditional Priors"][2]["Sigma"] = "Psi 6"

kH["Problem"]["Conditional Priors"][3]["Type"] = "Uniform"
kH["Problem"]["Conditional Priors"][3]["Minimum"] = 0.1
kH["Problem"]["Conditional Priors"][3]["Maximum"] = 15


kH["Variables"][0]["Name"] = "Psi 1"
kH["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
kH["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

kH["Variables"][1]["Name"] = "Psi 2"
kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][1]["Prior Distribution"]["Minimum"] = 1e-2
kH["Variables"][1]["Prior Distribution"]["Maximum"] = 10.0

kH["Variables"][2]["Name"] = "Psi 3"
kH["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][2]["Prior Distribution"]["Minimum"] = 30.0
kH["Variables"][2]["Prior Distribution"]["Maximum"] = 50.0

kH["Variables"][3]["Name"] = "Psi 4"
kH["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][3]["Prior Distribution"]["Minimum"] =  5.0
kH["Variables"][3]["Prior Distribution"]["Maximum"] = 15.0

kH["Variables"][4]["Name"] = "Psi 5"
kH["Variables"][4]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][4]["Prior Distribution"]["Minimum"] = -1.0
kH["Variables"][4]["Prior Distribution"]["Maximum"] = +1.0

kH["Variables"][5]["Name"] = "Psi 6"
kH["Variables"][5]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][5]["Prior Distribution"]["Minimum"] = 0.0
kH["Variables"][5]["Prior Distribution"]["Maximum"] = 1.0


kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000
kH["Solver"]["Termination Criteria"]["Max Generations"] = 500

kH["General"]["Console Output"]["Verbosity"] = "Detailed"

kH.run()
