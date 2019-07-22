#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from stat000_helpers import *

import korali
k = korali.initialize()

k["Problem"]["Type"] = "Bayesian Inference"
k["Problem"]["Likelihood"]["Model"] = "Custom"

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000

k.setModel( evaluateModel )
k.dry()

###############################################################################

# Test Configuration

assert_value( k["Solver"]["Covariance Scaling"], 0.04 )
assert_value( k["Solver"]["Burn In Default"], 0 )
assert_value( k["Solver"]["Max Rho Update"], 1.0 )
assert_value( k["Solver"]["Min Rho Update"], 1e-5 )
assert_value( k["Solver"]["Population Size"], 5000 )
assert_boolean( k["Solver"]["Use Local Covariance"], False )

# Test Internals

assert_value( k["Solver"]["Internal"]["Accepted Samples Count"], 5000 )
assert_value( k["Solver"]["Internal"]["Annealing Exponent"], 0.0 )
assert_value( k["Solver"]["Internal"]["Coefficient of Variation"], 0.0 )
assert_value( k["Solver"]["Internal"]["Covariance Matrix"][0], 0.0 )
assert_value( k["Solver"]["Internal"]["Proposals Acceptance Rate"], 1.0 )
assert_value( k["Solver"]["Internal"]["Selection Acceptance Rate"], 1.0 )

# Test Variables

assert_string( k["Variables"][0]["Prior Distribution"]["Type"], "Uniform" )
assert_value( k["Variables"][0]["Prior Distribution"]["Maximum"], 10 )
assert_value( k["Variables"][0]["Prior Distribution"]["Minimum"], -10 )

