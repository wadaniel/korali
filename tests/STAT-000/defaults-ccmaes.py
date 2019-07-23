#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from stat000_helpers import *

from math import isclose

import korali
k = korali.initialize()
k.setModel(evaluateModel)
k.addConstraint( g1 )

k["Problem"]["Type"] = "Constrained Optimization"
k["Problem"]["Objective"] = "Maximize"

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["Lower Bound"] = -10.0;
k["Variables"][0]["Upper Bound"] = +10.0;

k["Solver"]["Type"]  = "CCMAES" 
k["Solver"]["Sample Count"] = 32

k.dry()

###############################################################################

# Testing Configuration

assert_value( k["Solver"]["Covariance Matrix Adaption Strength"], 0.1 )
assert_value( k["Solver"]["Global Success Learning Rate"], 0.2 )
assert_value( k["Solver"]["Initial Cumulative Covariance"], -1.0 )
assert_value( k["Solver"]["Initial Damp Factor"], -1.0 )
assert_value( k["Solver"]["Initial Sigma Cumulation Factor"], -1.0 )
assert_boolean( k["Solver"]["Is Sigma Bounded"], False )
assert_value( k["Solver"]["Max Covariance Matrix Corrections"], 1e6 )
assert_string( k["Solver"]["Mu Type"], "Logarithmic" )
assert_value( k["Solver"]["Mu Value"], 16 )
assert_value( k["Solver"]["Normal Vector Learning Rate"], 0.3333333333333333 )
assert_value( k["Solver"]["Sample Count"], 32 )
assert_value( k["Solver"]["Target Success Rate"], 0.1818 )

# Testing Internals

assert_value( k["Solver"]["Internal"]["Chi Square Number"], 0.7976190476190477 )
assert_value( k["Solver"]["Internal"]["Chi Square Number Discrete Mutations"], 0.7976190476190477 )
assert_value( k["Solver"]["Internal"]["Covariance Matrix Adaption Factor"],  0.03333333333333333 )
assert_value( k["Solver"]["Internal"]["Cumulative Covariance"], 0.7142857142857143 )
assert_value( k["Solver"]["Internal"]["Current Sample Count"], 2 )
assert_value( k["Solver"]["Internal"]["Current Sample Mu"], 1 )
assert_value( k["Solver"]["Internal"]["Damp Factor"], 1.5 )
assert_value( k["Solver"]["Internal"]["Effective Mu"], 1.0 )
assert_value( k["Solver"]["Internal"]["Global Success Rate"], 0.5 )
assert_value( k["Solver"]["Internal"]["Sigma"], 5.0 )
assert_value( k["Solver"]["Internal"]["Trace"], 25.0 )
assert_boolean( k["Solver"]["Internal"]["Is Viability Regime"], True )
assert_value( k["Solver"]["Internal"]["Granularity"][0], 0.0 )
assert_value( k["Solver"]["Internal"]["Number Discrete Mutations"], 0 )
assert_value( k["Solver"]["Internal"]["Number Masking Matrix Entries"], 0 )

# Testing Termination Criteria

assert_value( k["Solver"]["Termination Criteria"]["Max Condition Covariance Matrix"]["Value"], 1e18 )
assert_boolean( k["Solver"]["Termination Criteria"]["Max Fitness"]["Enabled"], False )
assert_value( k["Solver"]["Termination Criteria"]["Max Infeasible Resampling"]["Value"], 1e9 )
assert_value( k["Solver"]["Termination Criteria"]["Max Standard Deviation"]["Value"], 1e18)
assert_boolean( k["Solver"]["Termination Criteria"]["Min Fitness"]["Enabled"], False )
assert_value( k["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"], 1e-9 )
assert_value( k["Solver"]["Termination Criteria"]["Min Standard Deviation"]["Value"], 1e-12 )
assert_value( k["Solver"]["Termination Criteria"]["Min Standard Deviation Step Factor"]["Value"], 1e-18 )
assert_value( k["Solver"]["Viability Mu"], 1 )
assert_value( k["Solver"]["Viability Sample Count"], 2 )


# Testing Variables
assert_value( k["Variables"][0]["Initial Mean"], 0.0 )
assert_value( k["Variables"][0]["Initial Standard Deviation"], 5.0 )
assert_value( k["Variables"][0]["Lower Bound"], -10.0 )
assert_value( k["Variables"][0]["Minimum Standard Deviation Changes"], 0.0 )
assert_value( k["Variables"][0]["Upper Bound"], 10.0 )


