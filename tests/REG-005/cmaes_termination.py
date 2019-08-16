#!/bin/bash
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# Set Up CMAES run
#################################################


def run_cmaes(criteria, value):

    print("[Korali] Prepare CMAES run with Termination Criteria '{0}'".format(criteria))

    k = korali.initialize()

    k.setDirectModel(evaluateModel)

    k["Problem"]["Type"] = "Optimization"
    k["Problem"]["Objective"] = "Maximize"

    k["Problem"]["Variables"][0]["Name"] = "X"
    k["Problem"]["Variables"][0]["Lower Bound"] = -10.0
    k["Problem"]["Variables"][0]["Upper Bound"] = +10.0

    k["Solver"]["Type"] = "CMAES"
    k["Solver"]["Population Size"] = 32
    k["Solver"]["Termination Criteria"][criteria] = value

    k.run()

    if (criteria == "Max Generations"):
        assert_value(k["General"]["Current Generation"].getValue(), value)
        
    elif (criteria == "Max Model Evaluations"):
        assert_value(k["General"]["Function Evaluation Count"].getValue(), value)
    
    elif (criteria == "Max Infeasible Resamplings"):
        assert_greatereq(k["Solver"]["Internal"]["Infeasible Sample Count"].getValue(), value)
    
    elif (criteria == "Max Condition Covariance Matrix"):
        minEw = k["Solver"]["Internal"]["Minimum Covariance Eigenvalue"].getValue()
        maxEw = k["Solver"]["Internal"]["Maximum Covariance Eigenvalue"].getValue()
        assert_greatereq(maxEw/minEw, value)

    elif (criteria == "Max Value"):
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(), value)
  
    elif (criteria == "Min Value"):
        print("TODO: Min Val")
    
    elif (criteria == "Min Value Difference Threshold"):
        previous = k["Solver"]["Internal"]["Previous Best Ever Value"].getValue()
        current  = k["Solver"]["Internal"]["Best Ever Value"].getValue()
        assert_smallereq(previous-current, value)
 
    elif (criteria == "Min Standard Deviation"):
        assert_smallereq(k["Solver"]["Internal"]["Current Min Standard Deviation"].getValue(), value)
 
    elif (criteria == "Max Standard Deviation"):
        assert_greatereq(k["Solver"]["Internal"]["Current Max Standard Deviation"].getValue(), value)

    elif (criteria == "Min Standard Deviation Step Factor"):
        assert_smallereq(k["Solver"]["Internal"]["Current Min Standard Deviation Step"].getValue(), value)

    else:
        print("Termination Criteria not recognized!")
        exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cmaes_termination', description='Check Termination Criteria.')
    parser.add_argument('--criteria', help='Name of Termination Criteria', action='store', required = True)
    parser.add_argument('--value', help='Value of Termination Criteria', action='store', type = float, required = True)
    args = parser.parse_args()

    run_cmaes(args.criteria, args.value)



 
