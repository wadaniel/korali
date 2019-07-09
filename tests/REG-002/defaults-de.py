#!/usr/bin/env python3
import sys
sys.path.append('./helpers')

from reg002_helpers import *

import korali

k = korali.initialize()

k.setModel(evaluateModel)

k["Problem"] = "Direct Evaluation"
k["Solver"]  = "DE" 

k["Variables"][0]["Name"] = "X";
k["Variables"][0]["DE"]["Lower Bound"] = -10.0;
k["Variables"][0]["DE"]["Upper Bound"] = +10.0;

k["DE"]["Objective"] = "Maximize"
k["DE"]["Sample Count"] = 32
k["DE"]["Termination Criteria"]["Max Generations"]["Value"] = 0

k["Result Directory"] = "_defaults_de"

k.run()

###############################################################################

# Test Configuration

assert_string( k["DE"]["Accept Rule"] , "Greedy" )

assert_value( k["DE"]["Crossover Rate"] , 0.9 )

assert_boolean( k["DE"]["Fix Infeasible"] , True )

assert_value( k["DE"]["Mutation Rate"] , 0.5 )

assert_string( k["DE"]["Mutation Rule"] , "Default" )

assert_string( k["DE"]["Objective"] , "Maximize" )

assert_string( k["DE"]["Parent Selection Rule"] , "Random" )

assert_value( k["Console Output Frequency"] , 1 )

assert_value( k["File Output Frequency"] , 1 )


# Test Internal

assert_value( k["DE"]["Internal"]["Evaluation Sign"] , 1.0 ) 

assert_value( k["DE"]["Internal"]["Max Width"][0] , 0.0 ) 


# Test Termination Criteria

assert_boolean( k["DE"]["Termination Criteria"]["Max Fitness"]["Enabled"] , False ) 

assert_boolean( k["DE"]["Termination Criteria"]["Max Generations"]["Enabled"] , True ) 

assert_value( k["DE"]["Termination Criteria"]["Max Generations"]["Value"] , 0 ) 

assert_boolean( k["DE"]["Termination Criteria"]["Min Fitness"]["Enabled"] , False ) 

assert_boolean( k["DE"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Enabled"] , True ) 

assert_value( k["DE"]["Termination Criteria"]["Min Fitness Diff Threshold"]["Value"] , 1e-9 ) 

assert_boolean( k["DE"]["Termination Criteria"]["Min Step Size"]["Enabled"] , False ) 

assert_value( k["DE"]["Termination Criteria"]["Min Step Size"]["Value"] , 1e-12 ) 
