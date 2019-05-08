#!/usr/bin/env python3
import sys
sys.path.append('./model')
from ackley import ackley
import libkorali

korali = libkorali.Engine(ackley)

korali["Verbosity"] = "Normal";

for i in range(4):
 korali["Parameters"][i]["Name"] = "X" + str(i)
 korali["Parameters"][i]["Distribution"] = "Uniform"
 korali["Parameters"][i]["Type"] = "Computational"
 korali["Parameters"][i]["Minimum"] = -32.0
 korali["Parameters"][i]["Maximum"] = +32.0

korali["Problem"]["Objective"] = "Direct Evaluation";
korali["Solver"]["Method"] = "TMCMC"
korali["Solver"]["Covariance Scaling"] = 0.02
korali["Solver"]["Population Size"] = 5000
korali["Solver"]["Burn In"] = 5

korali.run()
