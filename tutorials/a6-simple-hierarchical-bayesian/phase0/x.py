#!/usr/bin/env python3
import sys
import numpy as np
sys.path.append('../setup/model')
sys.path.append('/setup/model')
from model import *
import korali

# N : number of datasets
N = 5 
# Ns: number os samples in each dataset
Ns = 100

k = korali.initialize()

k["Problem"]["Type"] = "Execution"
k["Problem"]["Execution Model"] = lambda modelData: normal_rnds(modelData, Ns)

k["Variables"][0]["Name"] = "V1"
k["Variables"][1]["Name"] = "V2"
k["Variables"][0]["Loaded Values"] = np.random.normal( 10, 4, N ).tolist()
k["Variables"][1]["Loaded Values"] = np.random.lognormal( 0, 1, N ).tolist()

k["Solver"]["Type"] = "Executor"
k["Solver"]["Executions Per Generation"] = 1
k["Console Output"]["Verbosity"] = "Detailed"

k.run()
