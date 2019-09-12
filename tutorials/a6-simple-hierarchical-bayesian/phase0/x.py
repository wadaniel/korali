#!/usr/bin/env python3
import sys
import numpy as np
sys.path.append('../setup/model')
sys.path.append('/setup/model')
from model import *
import korali

# N : number of datasets
N = 10
# Ns: number os samples in each dataset
Ns = 100

k = korali.initialize()

k["Problem"]["Type"] = "Propagation"

k["Problem"]["Variables"][0]["Name"] = "V1"
k["Problem"]["Variables"][1]["Name"] = "V2"

k["Problem"]["Variables"][0]["Loaded Values"] = np.random.normal( 10, 4, N )
k["Problem"]["Variables"][1]["Loaded Values"] = np.random.lognormal( 0, 1, N )

k["Solver"]["Type"] = "Propagator"

k["General"]["Console Output"]["Verbosity"] = "Detailed"

k.setNoresultModel(lambda modelData: normal_rnds(modelData, Ns))

k.run()
