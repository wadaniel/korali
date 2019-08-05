#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from helpers import *

from math import isclose

import korali
k = korali.initialize()
k.setDirectModel(evaluateModel)
k.addConstraint( g1 )

k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["Lower Bound"] = -10.0;
k["Variables"][0]["Upper Bound"] = +10.0;

k["Solver"]["Type"]  = "CMAES" 
k["Solver"]["Sample Count"] = 32

k.dry()

###############################################################################

# Testing Configuration

assert_value( k["Solver"]["Covariance Matrix Adaption Strength"], 0.1 )
assert_value( k["Solver"]["Global Success Learning Rate"], 0.2 )
assert_value( k["Solver"]["Initial Cumulative Covariance"], -1.0 )
assert_value( k["Solver"]["Initial Damp Factor"], -1.0 )
assert_value( k["Solver"]["Initial Sigma Cumulation Factor"], -1.0 )
assert_value( k["Solver"]["Is Sigma Bounded"], False )
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
assert_value( k["Solver"]["Internal"]["Cumulative Covariance"], 0.6894039888609658) 
assert_value( k["Solver"]["Internal"]["Current Sample Count"], 2 )
assert_value( k["Solver"]["Internal"]["Current Sample Mu"], 2)
assert_value( k["Solver"]["Internal"]["Damp Factor"], 1.5471460885333408 )
assert_value( k["Solver"]["Internal"]["Effective Mu"], 1.459789888852586 )
assert_value( k["Solver"]["Internal"]["Global Success Rate"], 0.5 )
assert_value( k["Solver"]["Internal"]["Sigma"], 6.0 )
assert_value( k["Solver"]["Internal"]["Trace"], 36.0 )
assert_value( k["Solver"]["Internal"]["Is Viability Regime"], True )
assert_value( k["Solver"]["Internal"]["Granularity"][0], 0.0 )
assert_value( k["Solver"]["Internal"]["Number Of Discrete Mutations"], 0 )
assert_value( k["Solver"]["Internal"]["Number Masking Matrix Entries"], 0 )

# Testing Variables
assert_value( k["Variables"][0]["Initial Mean"], 0.0 )
assert_value( k["Variables"][0]["Initial Standard Deviation"], 6.0 )
assert_value( k["Variables"][0]["Lower Bound"], -10.0 )
assert_value( k["Variables"][0]["Minimum Standard Deviation Changes"], 0.0 )
assert_value( k["Variables"][0]["Upper Bound"], 10.0 )


