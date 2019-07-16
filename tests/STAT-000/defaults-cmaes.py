#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from stat000_helpers import *

from math import isclose

import korali
k = korali.initialize()
k.setModel(evaluateModel)

k["Problem"] = "Direct Evaluation"
k["Solver"]  = "CMAES" 

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["CMAES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CMAES"]["Upper Bound"] = +10.0;

k["CMAES"]["Objective"] = "Maximize"
k["CMAES"]["Sample Count"] = 32
k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 0

k["Result Directory"] = "_defaults_cmaes"

k.run()

###############################################################################

# Testing Configuration

assert_value( k["CMAES"]["Covariance Matrix Adaption Strength"], 0.1 )

assert_value( k["CMAES"]["Global Success Learning Rate"], 0.2 )

assert_value( k["CMAES"]["Initial Cumulative Covariance"], -1.0 )

assert_value( k["CMAES"]["Initial Damp Factor"], -1.0 )

assert_value( k["CMAES"]["Initial Sigma Cumulation Factor"], -1.0 )

assert_boolean( k["CMAES"]["Is Sigma Bounded"], False )

assert_value( k["CMAES"]["Max Covariance Matrix Corrections"], 1e6 )

assert_string( k["CMAES"]["Mu Type"], "Logarithmic" )

assert_value( k["CMAES"]["Mu Value"], 16 )

assert_value( k["CMAES"]["Normal Vector Learning Rate"], -1.0 )

assert_string( k["CMAES"]["Objective"], "Maximize" )

assert_value( k["CMAES"]["Sample Count"], 32 )

assert_value( k["CMAES"]["Target Success Rate"], 0.1818 )


# Testing Internals

assert_value( k["CMAES"]["Internal"]["Chi Number"], 0.7976190476190477 )

assert_value( k["CMAES"]["Internal"]["Covariance Matrix Adaption Factor"], -1.0 )

assert_value( k["CMAES"]["Internal"]["Cumulative Covariance"], 0.564218767613317 )

assert_value( k["CMAES"]["Internal"]["Current Sample Count"], 32 )

assert_value( k["CMAES"]["Internal"]["Current Sample Mu"], 16 )

assert_value( k["CMAES"]["Internal"]["Damp Factor"], 3.892701587864561 )

assert_value( k["CMAES"]["Internal"]["Effective Mu"], 9.17882891362855 )

assert_value( k["CMAES"]["Internal"]["Evaluation Sign"], 1.0 )

assert_value( k["CMAES"]["Internal"]["Global Success Rate"], -1.0 )

assert_value( k["CMAES"]["Internal"]["Sigma"], 5.0 )

assert_value( k["CMAES"]["Internal"]["Trace"], 25.0 )

assert_boolean( k["CMAES"]["Internal"]["Is Viability Regime"], False )


# Testing Termination Criteria

assert_value( k["CMAES"]["Termination Criteria"]["Max Condition Covariance Matrix"]["Value"], 1e18 )

assert_boolean( k["CMAES"]["Termination Criteria"]["Max Fitness"]["Enabled"], False )

assert_value( k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"], 0 )

assert_value( k["CMAES"]["Termination Criteria"]["Max Infeasible Resampling"]["Value"], 1e9 )

assert_value( k["CMAES"]["Termination Criteria"]["Max Standard Deviation"]["Value"], 1e18)

assert_boolean( k["CMAES"]["Termination Criteria"]["Min Fitness"]["Enabled"], False )

assert_value( k["CMAES"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"], 1e-9 )

assert_value( k["CMAES"]["Termination Criteria"]["Min Standard Deviation"]["Value"], 1e-12 )

assert_value( k["CMAES"]["Termination Criteria"]["Min Standard Deviation Step Factor"]["Value"], 1e-18 )

assert_value( k["CMAES"]["Viability Mu"], 1 )

assert_value( k["CMAES"]["Viability Sample Count"], 2 )


# Testing Variables

assert_value( k["Variables"][0]["CMAES"]["Initial Mean"], 0.0 )

assert_value( k["Variables"][0]["CMAES"]["Initial Standard Deviation"], 5.0 )

assert_value( k["Variables"][0]["CMAES"]["Lower Bound"], -10.0 )

assert_value( k["Variables"][0]["CMAES"]["Minimum Standard Deviation Changes"], 0.0 )

assert_value( k["Variables"][0]["CMAES"]["Upper Bound"], 10.0 )


