#!/usr/bin/env python3
import os
import sys
import math
sys.path.append('./model/')
from model import *
import numpy as np

### With Predetermined values

# Creating new experiment
import korali
e = korali.Experiment()

e["Random Seed"] = 0xC0FF33
e["Problem"]["Type"] = "Integration"
e["Problem"]["Integrand"] = pcubic

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0

e["Solver"]["Type"] = "Integrator/MonteCarlo"
e["Solver"]["Number Of Samples"] = 1000
e["Solver"]["Executions Per Generation"] = 10

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
  
# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

# Verify result
assert math.isclose(e["Results"]["Integral"], 0.25, rel_tol=0., abs_tol=0.05), f'Expected 0.25 is {e["Results"]["Integral"]}'
