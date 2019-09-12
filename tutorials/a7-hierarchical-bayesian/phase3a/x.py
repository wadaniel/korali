#!/usr/bin/env python3

# Importing computational model
import korali

# Creating hierarchical Bayesian problem from previous two problems
k = korali.initialize()

k["Problem"]["Type"]  = "Hierarchical Bayesian (Theta New)"

k["Problem"]["Psi Problem"] = '../setup/results_phase_2/final.json'

k["Problem"]["Variables"][0]["Name"] = "C1"
k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = 280.0
k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = 320.0

k["Problem"]["Variables"][1]["Name"] = "C2"
k["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] = 10.0
k["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = 70.0

k["Problem"]["Variables"][2]["Name"] = "C3"
k["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = 0.0
k["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = 5.0

k["Problem"]["Variables"][3]["Name"] = "Sigma"
k["Problem"]["Variables"][3]["Type"] = "Statistical"
k["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = 0
k["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = 20

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Target Coefficient Of Variation"] = 0.6
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Default Burn In"] = 2;
k["Solver"]["Max Chain Length"] = 1;
# k["Solver"]["Termination Criteria"]["Max Generations"] = 3;

k["General"]["Console Output"]["Verbosity"] = "Detailed"
k["General"]["Results Output"]["Path"] = "../setup/results_phase_3a/"

k.run()
