#!/usr/bin/env python3
import sys
import threading
sys.path.append('./model')
from ackley import m_ackley
from gaussian import gaussian
from eggholder import m_eggholder
from levi13 import m_levi13
from rosenbrock import m_rosenbrock
from shekel import m_shekel

import libkorali

error = """\nInvalid Args! Usage:\n
            \t./maximize_function 1 (=maximize -ackley)\n
            \t./mazimize_function 2 (=mazimize -eggholder)\n
            \t./mazimize_function 3 (=mazimize +gaussian)\n                   
            \t./mazimize_function 4 (=mazimize -levi13)\n                       
            \t./mazimize_function 5 (=mazimzie -rosenbrock)\n                   
            \t./mazimize_function 6 (=maximize -shekel)\n\n"""

if len(sys.argv) != 2:
    print(error)
    exit(1)

case = int(sys.argv[1])

if (case == 1) :
    nParams = 4
    lower, upper = -32.0, 32.0
    korali = libkorali.Engine(m_ackley)

elif (case == 2) :
    nParams = 2
    lower, upper = -512.0, 512.0
    korali = libkorali.Engine(m_eggholder)

elif (case == 3) :
    nParams = 5
    lower, upper = -32.0, 32.0
    korali = libkorali.Engine(gaussian)

elif (case == 4) :
    nParams = 2
    lower, upper = -10.0, 10.0
    korali = libkorali.Engine(m_levi13)

elif (case == 5) :
    nParams = 2
    lower, upper = -32.0, 32.0
    korali = libkorali.Engine(m_rosenbrock)

elif (case == 6) :
    nParams = 4
    lower, upper = 0.0, 10.0
    korali = libkorali.Engine(m_shekel)

else :
    print(error)
    exit(1)

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

korali.run()
