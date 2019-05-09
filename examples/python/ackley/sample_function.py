#!/usr/bin/env python3
import sys
sys.path.append('./model')
from ackley import ackley
import libkorali

korali = libkorali.Engine(ackley)

korali["Verbosity"] = "Normal";

korali["Problem"]["Type"] = "Direct";
for i in range(4):
 korali["Problem"]["Variables"][i]["Name"] = "X" + str(i)
 korali["Problem"]["Variables"][i]["Distribution"] = "Uniform"
 korali["Problem"]["Variables"][i]["Type"] = "Computational"
 korali["Problem"]["Variables"][i]["Minimum"] = -32.0
 korali["Problem"]["Variables"][i]["Maximum"] = +32.0



korali["Solver"]["Method"] = "TMCMC"
korali["Solver"]["Covariance Scaling"] = 0.02
korali["Solver"]["Population Size"] = 1000
korali["Solver"]["Burn In"] = 5

korali.run()
