#!/usr/bin/env python3
import sys
import threading
sys.path.append('./model')
from ackley import m_ackley

import libkorali

nParams = 4
lower, upper = -32.0, 32.0
korali = libkorali.Engine()

korali["Seed"]      = 0xC0FFEE
korali["Verbosity"] = "Detailed"

for i in range(nParams):
    korali["Problem"]["Variables"][i]["Name"] = "X" + str(i)
    korali["Problem"]["Variables"][i]["Type"] = "Computational"
    korali["Problem"]["Variables"][i]["Distribution"] = "Uniform"
    korali["Problem"]["Variables"][i]["Minimum"] = lower
    korali["Problem"]["Variables"][i]["Maximum"] = upper
     
korali["Problem"]["Type"] = "Direct"

korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Lambda"] = 10
korali["Solver"]["Termination Criteria"]["Max Generations"] = 100
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-12

korali.run(m_ackley)
