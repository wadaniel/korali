#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from reg002_helpers import *

import korali
k = korali.initialize()
k.setLikelihood( evaluateModel )

k["Problem"] = "Bayesian"
k["Solver"] = "TMCMC"

k["Bayesian"]["Likelihood"]["Type"] = "Direct"

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -10.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +10.0

k["TMCMC"]["Population Size"] = 5000
#k["Termination Criteria"]["Max Generations"]["Value"] = 1

k["Result Directory"] = "_a2_sampling_tmcmc"

k.run()

###############################################################################

# Test Configuration

# Test Internals

# Test Termination Criteria

# Test Variables


