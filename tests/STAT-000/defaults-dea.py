#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from helpers import *

import korali

k = korali.initialize()

k.setDirectModel(evaluateModel)

k["Problem"]["Type"] = "Optimization"
k["Problem"]["Objective"] = "Maximize"

k["Problem"]["Variables"][0]["Name"] = "X";
k["Problem"]["Variables"][0]["Lower Bound"] = -10.0;
k["Problem"]["Variables"][0]["Upper Bound"] = +10.0;

k["Solver"]["Type"] = "DEA"
k["Solver"]["Population Size"] = 32

k.dry()

###############################################################################

# Test Configuration

assert_string( k["Solver"]["Accept Rule"] , "Greedy" )
assert_value( k["Solver"]["Crossover Rate"] , 0.9 )
assert_value( k["Solver"]["Fix Infeasible"] , True )
assert_value( k["Solver"]["Mutation Rate"] , 0.5 )
assert_string( k["Solver"]["Mutation Rule"] , "Fixed" )
assert_string( k["Solver"]["Parent Selection Rule"] , "Random" )

# Test Internal

assert_value( k["Solver"]["Internal"]["Max Distances"][0] , 0.0 ) 

