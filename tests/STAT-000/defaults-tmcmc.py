#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from helpers import *

import korali
k = korali.initialize()

k["Problem"]["Type"] = "Bayesian Inference"
k["Problem"]["Likelihood"]["Model"] = "Custom"

k["Problem"]["Variables"][0]["Name"] = "X"
k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000

k.setLikelihoodModel(evaluateModel)
k.dry()

###############################################################################

# Test Configuration

assert_value( k["Solver"]["Covariance Scaling"], 0.04 )
assert_value( k["Solver"]["Default Burn In"], 0 )
assert_value( k["Solver"]["Max Annealing Exponent Update"], 1.0 )
assert_value( k["Solver"]["Min Annealing Exponent Update"], 1e-5 )
assert_value( k["Solver"]["Population Size"], 5000 )
assert_value( k["Solver"]["Use Local Covariance"], False )

# Test Internals

assert_value( k["Solver"]["Internal"]["Accepted Samples Count"], 5000 )
assert_value( k["Solver"]["Internal"]["Annealing Exponent"], 0.0 )
assert_value( k["Solver"]["Internal"]["Coefficient Of Variation"], 0.0 )
assert_value( k["Solver"]["Internal"]["Covariance Matrix"][0], 0.0 )
assert_value( k["Solver"]["Internal"]["Proposals Acceptance Rate"], 1.0 )
assert_value( k["Solver"]["Internal"]["Selection Acceptance Rate"], 1.0 )

# Test Variables

assert_string( k["Problem"]["Variables"][0]["Prior Distribution"]["Type"], "Uniform" )
assert_value( k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"], 10 )
assert_value( k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"], -10 )

