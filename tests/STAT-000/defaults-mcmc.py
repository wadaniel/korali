#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from helpers import *

import korali
k = korali.initialize()

k.setDirectModel(evaluateModel)

k["Problem"]["Type"] = "Sampling"

k["Problem"]["Variables"][0]["Name"] = "X"
k["Problem"]["Variables"][0]["Initial Mean"] = 0.0
k["Problem"]["Variables"][0]["Initial Standard Deviation"] = 1.000

k["Solver"]["Type"]  = "MCMC"

k["General"]["Console Output"]["Frequency"] = 500
k["General"]["Results Output"]["Frequency"] = 500

k.dry()

###############################################################################

# Test General Configuration

assert_value( k["General"]["Console Output"]["Frequency"], 500 )
assert_value( k["General"]["Results Output"]["Frequency"], 500 )

# Test Solver Configuration
assert_value( k["Solver"]["Burn In"], 0 )
assert_value( k["Solver"]["Chain Covariance Scaling"], 1.0 )
assert_value( k["Solver"]["Non Adaption Period"], 0 )
assert_value( k["Solver"]["Rejection Levels"], 1 )
assert_value( k["Solver"]["Use Adaptive Sampling"], False)

# Test Internals

assert_value( k["Solver"]["Internal"]["Chain Length"], 0.0 )
assert_value( k["Solver"]["Internal"]["Cholesky Decomposition Covariance"][0], 1.0 )
assert_value( k["Solver"]["Internal"]["Rejection Alphas"][0], 0.0 )

# Test Variables

assert_value( k["Problem"]["Variables"][0]["Initial Mean"], 0.0 )
assert_value( k["Problem"]["Variables"][0]["Initial Standard Deviation"], 1.0 )


