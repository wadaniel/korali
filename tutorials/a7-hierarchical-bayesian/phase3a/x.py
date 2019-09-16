#!/usr/bin/env python3

# Importing computational model
import korali

# Creating hierarchical Bayesian problem from previous two problems
k = korali.initialize()

k["Problem"]["Type"]  = "Hierarchical Bayesian (Theta New)"

k["Problem"]["Psi Problem Path"] = '../setup/results_phase_2/final.json'

k["Variables"][0]["Name"] = "C1"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

k["Variables"][1]["Name"] = "C2"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = 10.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = 70.0

k["Variables"][2]["Name"] = "C3"
k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][2]["Prior Distribution"]["Maximum"] = 5.0

k["Variables"][3]["Name"] = "Sigma"
k["Variables"][3]["Bayesian Type"] = "Statistical"
k["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][3]["Prior Distribution"]["Minimum"] = 0
k["Variables"][3]["Prior Distribution"]["Maximum"] = 20

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 2;
k["Solver"]["Max Chain Length"] = 1;
# k["Solver"]["Termination Criteria"]["Max Generations"] = 3;

k["Console Output"]["Verbosity"] = "Detailed"
k["Results Output"]["Path"] = "../setup/results_phase_3a/"

k.run()
