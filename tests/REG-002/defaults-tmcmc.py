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
k["Termination Criteria"]["Max Generations"] = 0

k["Result Directory"] = "_defaults_tmcmc"

k.run()

###############################################################################

# Test Configuration

assert_value( k["File Output Frequency"], 1 )

assert_value( k["TMCMC"]["Covariance Scaling"], 0.04 )

assert_value( k["TMCMC"]["Burn In"], 0 )

assert_value( k["TMCMC"]["Max Rho Update"], 1.0 )

assert_value( k["TMCMC"]["Min Rho Update"], 1e-5 )

assert_value( k["TMCMC"]["Population Size"], 5000 )

assert_boolean( k["TMCMC"]["Use Local Covariance"], False )


# Test Internals

assert_value( k["TMCMC"]["Internal"]["Accepted Samples Count"], 5000 ) # what is this (DW)

assert_value( k["TMCMC"]["Internal"]["Annealing Exponent"], 0.0 )

assert_value( k["TMCMC"]["Internal"]["Coefficient of Variation"], 0.0 )

assert_value( k["TMCMC"]["Internal"]["Covariance Matrix"][0], 0.0 )

assert_value( k["TMCMC"]["Internal"]["Proposals Acceptance Rate"], 1.0 )

assert_value( k["TMCMC"]["Internal"]["Selection Acceptance Rate"], 1.0 )


# Test Termination Criteria

assert_value( k["Termination Criteria"]["Max Function Evaluations"], 5e7 ) 
#TODO should be k["MCMC"]["Termination Criteria"]["Max .. "]["Value"] = ..

assert_value( k["Termination Criteria"]["Max Generations"], 0 )
#TODO should be k["MCMC"]["Termination Criteria"]["Max .. "]["Value"] = ..


# Test Variables

assert_value( k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"], 10 )

assert_value( k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"], -10 )

assert_string( k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"], "Uniform" )
