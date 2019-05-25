#!/usr/bin/env python3
import sys
sys.path.append('./model')
from ackley import m_ackley
import korali

k = korali.Engine()

k["Verbosity"] = "Normal";

k["Problem"]["Type"] = "Direct";
for i in range(4):
    k["Problem"]["Variables"][i]["Name"] = "X" + str(i)
    k["Problem"]["Variables"][i]["Distribution"] = "Uniform"
    k["Problem"]["Variables"][i]["Type"] = "Computational"
    k["Problem"]["Variables"][i]["Minimum"] = -32.0
    k["Problem"]["Variables"][i]["Maximum"] = +32.0

k["Solver"]["Method"] = "TMCMC"
k["Solver"]["Covariance Scaling"] = 0.02
k["Solver"]["Population Size"] = 1000
k["Solver"]["Burn In"] = 5

k.run(m_ackley)
