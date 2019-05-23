#!/usr/bin/env python3
import sys
sys.path.append('./model')
from ackley import m_ackley
import libkorali

korali = libkorali.Engine()

korali["Seed"]      = 0xC0FFEE
korali["Verbosity"] = "Detailed"
korali["Problem"]["Type"] = "Direct"

for i in range(4):
 korali["Problem"]["Variables"][i]["Name"] = "X" + str(i)
 korali["Problem"]["Variables"][i]["Type"] = "Computational"
 korali["Problem"]["Variables"][i]["Distribution"] = "Uniform"
 korali["Problem"]["Variables"][i]["Minimum"] = -32.0
 korali["Problem"]["Variables"][i]["Maximum"] = +32.0

korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Lambda"] = 10
korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-12

korali.run(m_ackley)
