#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from stat000_helpers import *

from math import isclose

import korali
k = korali.initialize()
k.setModel(evaluateModel)
k.addConstraint( g1 )

k["Problem"] = "Constrained Evaluation"
k["Solver"]  = "CCMAES" 

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["CCMAES"]["Lower Bound"] = -10.0;
k["Variables"][0]["CCMAES"]["Upper Bound"] = +10.0;

k["CCMAES"]["Objective"] = "Maximize"
k["CCMAES"]["Sample Count"] = 32
k["CCMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 0

k["Result Directory"] = "_defaults_ccmaes"

k.run()

###############################################################################

# Testing Configuration

assert_value( k["CCMAES"]["Covariance Matrix Adaption Strength"], 0.1 )

assert_value( k["CCMAES"]["Global Success Learning Rate"], 0.2 )

assert_value( k["CCMAES"]["Initial Cumulative Covariance"], -1.0 )

assert_value( k["CCMAES"]["Initial Damp Factor"], -1.0 )

assert_value( k["CCMAES"]["Initial Sigma Cumulation Factor"], -1.0 )

assert_boolean( k["CCMAES"]["Is Sigma Bounded"], False )

assert_value( k["CCMAES"]["Max Covariance Matrix Corrections"], 1e6 )

assert_string( k["CCMAES"]["Mu Type"], "Logarithmic" )

assert_value( k["CCMAES"]["Mu Value"], 16 )

assert_value( k["CCMAES"]["Normal Vector Learning Rate"], 0.3333333333333333 )

assert_string( k["CCMAES"]["Objective"], "Maximize" )

assert_value( k["CCMAES"]["Sample Count"], 32 )

assert_value( k["CCMAES"]["Target Success Rate"], 0.1818 )


# Testing Internals

assert_value( k["CCMAES"]["Internal"]["Chi Number"], 0.7976190476190477 )

assert_value( k["CCMAES"]["Internal"]["Covariance Matrix Adaption Factor"],  0.03333333333333333 )

assert_value( k["CCMAES"]["Internal"]["Cumulative Covariance"], 0.7142857142857143 )

assert_value( k["CCMAES"]["Internal"]["Current Sample Count"], 2 )

assert_value( k["CCMAES"]["Internal"]["Current Sample Mu"], 1 )

assert_value( k["CCMAES"]["Internal"]["Damp Factor"], 1.5 )

assert_value( k["CCMAES"]["Internal"]["Effective Mu"], 1.0 )

assert_value( k["CCMAES"]["Internal"]["Evaluation Sign"], 1.0 )

assert_value( k["CCMAES"]["Internal"]["Global Success Rate"], 0.5 )

assert_value( k["CCMAES"]["Internal"]["Sigma"], 5.0 )

assert_value( k["CCMAES"]["Internal"]["Trace"], 25.0 )

assert_boolean( k["CCMAES"]["Internal"]["Is Viability Regime"], True )


# Testing Termination Criteria

assert_value( k["CCMAES"]["Termination Criteria"]["Max Condition Covariance Matrix"]["Value"], 1e18 )

assert_boolean( k["CCMAES"]["Termination Criteria"]["Max Fitness"]["Enabled"], False )

assert_value( k["CCMAES"]["Termination Criteria"]["Max Generations"]["Value"], 0 )

assert_value( k["CCMAES"]["Termination Criteria"]["Max Infeasible Resampling"]["Value"], 1e9 )

assert_value( k["CCMAES"]["Termination Criteria"]["Max Standard Deviation"]["Value"], 1e18)

assert_boolean( k["CCMAES"]["Termination Criteria"]["Min Fitness"]["Enabled"], False )

assert_value( k["CCMAES"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"], 1e-9 )

assert_value( k["CCMAES"]["Termination Criteria"]["Min Standard Deviation"]["Value"], 1e-12 )

assert_value( k["CCMAES"]["Termination Criteria"]["Min Standard Deviation Step Factor"]["Value"], 1e-18 )

assert_value( k["CCMAES"]["Viability Mu"], 1 )

assert_value( k["CCMAES"]["Viability Sample Count"], 2 )


# Testing Variables

assert_value( k["Variables"][0]["CCMAES"]["Initial Mean"], 0.0 )

assert_value( k["Variables"][0]["CCMAES"]["Initial Standard Deviation"], 5.0 )

assert_value( k["Variables"][0]["CCMAES"]["Lower Bound"], -10.0 )

assert_value( k["Variables"][0]["CCMAES"]["Minimum Standard Deviation Changes"], 0.0 )

assert_value( k["Variables"][0]["CCMAES"]["Upper Bound"], 10.0 )


