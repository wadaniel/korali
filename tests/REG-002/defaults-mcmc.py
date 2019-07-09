#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from reg002_helpers import *

import korali
k = korali.initialize()

k.setModel(evaluateModel)

k["Problem"] = "Direct Evaluation"
k["Solver"]  = "MCMC"

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["MCMC"]["Initial Mean"] = 0.0
k["Variables"][0]["MCMC"]["Standard Deviation"] = 1.000

k["MCMC"]["Max Chain Length"] = 0
k["MCMC"]["Use Adaptive Sampling"] = True

k["Console Output Frequency"] = 500
k["File Output Frequency"] = 500

k["Result Directory"] = "_defaults_mcmc"

k.run()
