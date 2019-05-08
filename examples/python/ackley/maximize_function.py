#!/usr/bin/env python3
import sys
import threading
sys.path.append('./model')
from ackley import ackley
import libkorali

korali = libkorali.Engine(ackley)

korali["Verbosity"] = "Normal"

for i in range(4):
 korali["Parameters"][i]["Name"] = "X" + str(i)
 korali["Parameters"][i]["Distribution"] = "Uniform"
 korali["Parameters"][i]["Type"] = "Computational"
 korali["Parameters"][i]["Minimum"] = -32.0
 korali["Parameters"][i]["Maximum"] = +32.0

korali["Problem"]["Type"] = "Direct"

korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11
korali["Solver"]["Termination Criteria"]["Max Generations"] = 600
korali["Solver"]["Lambda"] = 128

korali.run()
