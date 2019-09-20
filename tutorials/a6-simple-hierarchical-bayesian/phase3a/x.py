#!/usr/bin/env python3

# Importing computational model
import korali

# Creating hierarchical Bayesian problem from previous two problems
k = korali.initialize()

k["Problem"]["Type"]  = "Hierarchical Bayesian (Theta New)"
k["Problem"]["Psi Problem Path"] = '../setup/results_phase_2/final.json'

k["Variables"][0]["Name"] = "mu"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] =  -20.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 40.0

k["Variables"][1]["Name"] = "sigma"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 10.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 1;

k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = 8 

k["Console Output"]["Verbosity"] = "Detailed"
k["Results Output"]["Path"] = "../setup/results_phase_3a/"

k.run()
