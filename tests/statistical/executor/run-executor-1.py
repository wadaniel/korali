#!/usr/bin/env python3
import os
import sys
sys.path.append('./model/')
from model import model_propagation
import numpy as np

### With Precomputed values

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Propagation"
e["Problem"]["Execution Model"] = model_propagation

e["Variables"][0]["Name"] = "Mean"
e["Variables"][0]["Precomputed Values"] = list(range(-50, 50))
e["Variables"][1]["Name"] = "Variance"
e["Variables"][1]["Precomputed Values"] = list(range(50, 150))

e["Solver"]["Type"] = "Executor"
e["Solver"]["Executions Per Generation"] = 5

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

