#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Checking that subproblem data exists 
if(not os.path.isdir('../data')):
 os.system('./generateData.sh')

# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"]  = "Hierarchical Bayesian"
kH["Problem"]["Model"] = "Sample Psi"
kH["Problem"]["Sub Problems"][0] = korali.getResults('../data/000/final.json')
kH["Problem"]["Sub Problems"][1] = korali.getResults('../data/001/final.json')
kH["Problem"]["Sub Problems"][2] = korali.getResults('../data/002/final.json')
kH["Problem"]["Sub Problems"][3] = korali.getResults('../data/003/final.json')
kH["Problem"]["Sub Problems"][4] = korali.getResults('../data/004/final.json')

# Add probability of theta given psi
kH["Problem"]["Conditional Priors"][0]["Type"] = "Gaussian"
kH["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1"
kH["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2"

kH["Problem"]["Conditional Priors"][1]["Type"] = "Uniform"
kH["Problem"]["Conditional Priors"][1]["Minimum"] = 1.0
kH["Problem"]["Conditional Priors"][1]["Maximum"] = "Psi 3"

kH["Variables"][0]["Name"] = "Psi 1"
kH["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

kH["Variables"][1]["Name"] = "Psi 2"
kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][1]["Prior Distribution"]["Minimum"] =   0.0
kH["Variables"][1]["Prior Distribution"]["Maximum"] = +10.0

kH["Variables"][2]["Name"] = "Psi 3"
kH["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][2]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][2]["Prior Distribution"]["Maximum"] = +10.0

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000
kH["Solver"]["Termination Criteria"]["Max Generations"] = 100

kH.run()
