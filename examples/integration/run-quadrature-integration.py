#!/usr/bin/env python3
import sys
sys.path.append('./_model/')
from integrand import *

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Integration"
e["Problem"]["Integrand"] = lambda modelData: integrand(modelData)

e["Variables"][0]["Name"] = "x"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0
e["Variables"][0]["Number Of Gridpoints"] = 10

e["Variables"][1]["Name"] = "y"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = 1.0
e["Variables"][1]["Number Of Gridpoints"] = 10

e["Variables"][2]["Name"] = "z"
e["Variables"][2]["Lower Bound"] = 0.0
e["Variables"][2]["Upper Bound"] = 1.0
e["Variables"][2]["Number Of Gridpoints"] = 10

e["Solver"]["Type"] = "Integrator/Quadrature"
e["Solver"]["Method"] = "Simpson"
e["Solver"]["Executions Per Generation"] = 100

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
