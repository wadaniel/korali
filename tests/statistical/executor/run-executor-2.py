#!/usr/bin/env python3
import os
import sys
sys.path.append('./model/')
from model import model_propagation
import numpy as np

### With Random values

# Creating new experiment
import korali
e = korali.Experiment()

e["Problem"]["Type"] = "Propagation"
e["Problem"]["Execution Model"] = model_propagation
e["Problem"]["Number Of Samples"] = 10

e["Variables"][0]["Name"] = "Mean"
e["Variables"][0]["Prior Distribution"] = "Normal 0"
e["Variables"][1]["Name"] = "Variance"
e["Variables"][1]["Prior Distribution"] = "Uniform 0"

e['Distributions'][0]['Name'] = 'Normal 0'
e['Distributions'][0]['Type'] = 'Univariate/Normal'
e['Distributions'][0]['Mean'] = 0.0
e['Distributions'][0]['Standard Deviation'] = +5.0

e['Distributions'][1]['Name'] = 'Uniform 0'
e['Distributions'][1]['Type'] = 'Univariate/Uniform'
e['Distributions'][1]['Minimum'] = +5.0
e['Distributions'][1]['Maximum'] = +10.0

e["Solver"]["Type"] = "Executor"
e["Solver"]["Executions Per Generation"] = 5

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
