#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from reg002_helpers import *

import korali

k = korali.initialize()

k.setModel(evaluateModel)

k["Problem"] = "Direct Evaluation"
k["Solver"]  = "DEA" 

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["DEA"]["Lower Bound"] = -10.0;
k["Variables"][0]["DEA"]["Upper Bound"] = +10.0;

k["DEA"]["Objective"] = "Maximize"
k["DEA"]["Sample Count"] = 32
k["DEA"]["Termination Criteria"]["Max Generations"]["Value"] = 0

k["Result Directory"] = "_defaults_dea"

k.run()

###############################################################################

# Test Configuration

assert_string( k["DEA"]["Accept Rule"] , "Greedy" )

assert_value( k["DEA"]["Crossover Rate"] , 0.9 )

assert_boolean( k["DEA"]["Fix Infeasible"] , True )

assert_value( k["DEA"]["Mutation Rate"] , 0.5 )

assert_string( k["DEA"]["Mutation Rule"] , "Default" )

assert_string( k["DEA"]["Objective"] , "Maximize" )

assert_string( k["DEA"]["Parent Selection Rule"] , "Random" )

assert_value( k["Console Output Frequency"] , 1 )

assert_value( k["File Output Frequency"] , 1 )


# Test Internal

assert_value( k["DEA"]["Internal"]["Evaluation Sign"] , 1.0 ) 

assert_value( k["DEA"]["Internal"]["Max Width"][0] , 0.0 ) 


# Test Termination Criteria

assert_boolean( k["DEA"]["Termination Criteria"]["Max Fitness"]["Enabled"] , False ) 

assert_boolean( k["DEA"]["Termination Criteria"]["Max Generations"]["Enabled"] , True ) 

assert_value( k["DEA"]["Termination Criteria"]["Max Generations"]["Value"] , 0 ) 

assert_boolean( k["DEA"]["Termination Criteria"]["Min Fitness"]["Enabled"] , False ) 

assert_boolean( k["DEA"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Enabled"] , True ) 

assert_value( k["DEA"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"] , 1e-9 ) 

assert_boolean( k["DEA"]["Termination Criteria"]["Min Step Size"]["Enabled"] , False ) 

assert_value( k["DEA"]["Termination Criteria"]["Min Step Size"]["Value"] , 1e-12 ) 
