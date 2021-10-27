#!/usr/bin/env python3
import os
import sys
sys.path.append('./model/')
from model import model_integration
import numpy as np

### With Random values

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Integration"
e["Problem"]["Integrand"] = model_integration
e["Problem"]["Integration Method"] = "Rectangle"

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0
e["Variables"][0]["Number Of Gridpoints"] = 10
e["Variables"][0]["Sampling Distribution"] = "Uniform"
e["Variables"][0]["Distribution Index"] = 0

e['Distributions'][0]['Name'] = 'Normal 0'
e['Distributions'][0]['Type'] = 'Univariate/Normal'
e['Distributions'][0]['Mean'] = 0.0
e['Distributions'][0]['Standard Deviation'] = +5.0

e["Solver"]["Type"] = "Integrator"
e["Solver"]["Executions Per Generation"] = 10
e["Solver"]["Termination Criteria"]["Max Generations"] = 1000

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
