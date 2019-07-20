#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from stat000_helpers import *

import korali

k = korali.initialize()

k.setModel(evaluateModel)

k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["Lower Bound"] = -10.0;
k["Variables"][0]["Upper Bound"] = +10.0;

k["Solver"]["Type"] = "DEA"
k["Solver"]["Sample Count"] = 32

k.dry()

###############################################################################

# Test Configuration

assert_string( k["Solver"]["Accept Rule"] , "Greedy" )
assert_value( k["Solver"]["Crossover Rate"] , 0.9 )
assert_boolean( k["Solver"]["Fix Infeasible"] , True )
assert_value( k["Solver"]["Mutation Rate"] , 0.5 )
assert_string( k["Solver"]["Mutation Rule"] , "Default" )
assert_string( k["Solver"]["Parent Selection Rule"] , "Random" )

# Test Internal

assert_value( k["Solver"]["Internal"]["Max Width"][0] , 0.0 ) 

# Test Termination Criteria

assert_boolean( k["Solver"]["Termination Criteria"]["Max Fitness"]["Enabled"] , False ) 
assert_boolean( k["Solver"]["Termination Criteria"]["Min Fitness"]["Enabled"] , False ) 
assert_boolean( k["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Enabled"] , True ) 
assert_value( k["Solver"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"] , 1e-9 ) 
assert_boolean( k["Solver"]["Termination Criteria"]["Min Step Size"]["Enabled"] , False ) 
assert_value( k["Solver"]["Termination Criteria"]["Min Step Size"]["Value"] , 1e-12 ) 
