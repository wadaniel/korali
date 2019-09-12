#!/usr/bin/env python3
import sys
import numpy as np
sys.path.append('../setup/model')
from model import *

import korali

N = 10;


k = korali.initialize()

k["Problem"]["Type"] = "Propagation"


k["Problem"]["Variables"][0]["Name"] = "V1"
k["Problem"]["Variables"][1]["Name"] = "V2"
k["Problem"]["Variables"][2]["Name"] = "V3"
k["Problem"]["Variables"][3]["Name"] = "V4"

k["Problem"]["Variables"][0]["Loaded Values"] = 300*np.ones(N)
k["Problem"]["Variables"][1]["Loaded Values"] = np.random.normal( 40, 4, N )
k["Problem"]["Variables"][2]["Loaded Values"] = np.random.lognormal( 0, 0.5, N )
k["Problem"]["Variables"][3]["Loaded Values"] = 5*np.ones(N)

k["Solver"]["Type"] = "Propagator"

k["General"]["Console Output"]["Verbosity"] = "Detailed"

# Setting the model
k.setNoresultModel( lambda modelData: logistic_reference(modelData) )


k.run()
