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

###############################################################################

# Test Configuration

assert_value( k["Console Output Frequency"], 500 )

assert_value( k["File Output Frequency"], 500 )

assert_value( k["MCMC"]["Burn In"], 0 )

assert_value( k["MCMC"]["Chain Covariance Increment"], 0.001 )

assert_value( k["MCMC"]["Chain Covariance Scaling"], 0.0 )

assert_value( k["MCMC"]["Max Chain Length"], 0 ) # What is this? (DW)

assert_value( k["MCMC"]["Non Adaption Period"], 0 )

assert_value( k["MCMC"]["Rejection Levels"], 1 )

assert_boolean( k["MCMC"]["Use Adaptive Sampling"], True )


# Test Internals

assert_value( k["MCMC"]["Internal"]["Chain Covariance"][0], 0.0 )

assert_value( k["MCMC"]["Internal"]["Chain Length"], 0.0 )

assert_value( k["MCMC"]["Internal"]["Covariance Matrix"][0], 1.0 )

assert_value( k["MCMC"]["Internal"]["Rejection Alphas"][0], 0.0 )

# Test Termination Criteria

assert_value( k["Termination Criteria"]["Max Function Evaluations"], 5e7 )

assert_value( k["Termination Criteria"]["Max Generations"], 5e6 )


# Test Variables

assert_value( k["Variables"][0]["MCMC"]["Initial Mean"], 0.0 )

assert_value( k["Variables"][0]["MCMC"]["Standard Deviation"], 1.0 )


