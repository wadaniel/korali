#!/bin/bash
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# CMAES run method
#################################################

def run_cmaes_with_termination_criterion(criterion, value):

    print("[Korali] Prepare CMAES run with Termination Criteria "\
            "'{0}'".format(criterion))

    k = korali.initialize()

    k["Problem"]["Type"] = "Evaluation/Direct"
    k["Problem"]["Objective"] = "Maximize"
    k["Problem"]["Objective Function"] = evaluateModel

    k["Variables"][0]["Name"] = "X"
    k["Variables"][0]["Lower Bound"] = +1.0
    k["Variables"][0]["Upper Bound"] = +10.0

    k["Solver"]["Type"] = "Optimizer/CMAES"
    k["Solver"]["Population Size"] = 8
    k["Solver"]["Termination Criteria"][criterion] = value

    k["Results Output"]["Frequency"] = 1000
    k["Random Seed"] = 1337

    k.run()

    if (criterion == "Max Generations"):
        assert_value(k["Internal"]["Current Generation"], value)
        
    elif (criterion == "Max Infeasible Resamplings"):
        assert_greatereq(k["Solver"]["Internal"]["Infeasible Sample Count"], value)
    
    elif (criterion == "Max Condition Covariance Matrix"):
        minEw = k["Solver"]["Internal"]["Minimum Covariance Eigenvalue"]
        maxEw = k["Solver"]["Internal"]["Maximum Covariance Eigenvalue"]
        assert_greatereq(maxEw/minEw, value)

    elif (criterion == "Max Value"):
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"], value)
  
    elif (criterion == "Min Value Difference Threshold"):
        previous = k["Solver"]["Internal"]["Previous Best Ever Value"]
        current  = k["Solver"]["Internal"]["Best Ever Value"]
        assert_smallereq(previous-current, value)
 
    elif (criterion == "Min Standard Deviation"):
        assert_smallereq(k["Solver"]["Internal"]["Current Min Standard Deviation"], value)
 
    elif (criterion == "Max Standard Deviation"):
        assert_greatereq(k["Solver"]["Internal"]["Current Max Standard Deviation"], value)
 
    elif (criterion == "Min Value"):
        print("TODO: Min Val")
 
    else:
        print("Termination Criterion not recognized!")
        exit(-1)


#################################################
# Main (called from run_test.sh with args)
#################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cmaes_termination', description='Check Termination Criterion.')
    parser.add_argument('--criterion', help='Name of Termination Criterion', action='store', required = True)
    parser.add_argument('--value', help='Value of Termination Criterion', action='store', type = float, required = True)
    args = parser.parse_args()
    
    run_cmaes_with_termination_criterion(args.criterion, args.value)



 
