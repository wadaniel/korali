#!/bin/bash
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# CCMAES run method
#################################################

def run_ccmaes(constraint):

    print("[Korali] Prepare CMAES run with Termination Criteria "\
            "'{0}'".format(constraint))

    k = korali.initialize()

    k.setDirectModel(evaluateModel)

    k["Problem"]["Type"] = "Optimization"
    k["Problem"]["Objective"] = "Maximize"

    k["Problem"]["Variables"][0]["Name"] = "X"
    k["Problem"]["Variables"][0]["Lower Bound"] = +1.0
    k["Problem"]["Variables"][0]["Upper Bound"] = +10.0

    k["Solver"]["Type"] = "CMAES"
    k["Solver"]["Population Size"] = 8
    k["Solver"]["Termination Criteria"][constraint] = 1

    k["General"]["Results Output"]["Frequency"] = 1000
    k["General"]["Random Seed"] = 1337

    return
    k.run()

    if (constraint == "Max Generations"):
        assert_value(k["General"]["Current Generation"].getValue(), 1)

    elif (constraint == "Max Model Evaluations"):
        assert_greatereq(k["General"]["Model Evaluation Count"].getValue(), 1)

    elif (constraint == "Max Infeasible Resamplings"):
        assert_greatereq(k["Solver"]["Internal"]["Infeasible Sample Count"].getValue(), 1)

    elif (constraint == "Max Condition Covariance Matrix"):
        minEw = k["Solver"]["Internal"]["Minimum Covariance Eigenvalue"].getValue()
        maxEw = k["Solver"]["Internal"]["Maximum Covariance Eigenvalue"].getValue()
        assert_greatereq(maxEw/minEw, 1)

    elif (constraint == "Max Value"):
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),
                1)

    elif (constraint == "Min Value Difference Threshold"):
        previous = k["Solver"]["Internal"]["Previous Best Ever Value"].getValue()
        current  = k["Solver"]["Internal"]["Best Ever Value"].getValue()
        assert_smallereq(previous-current, 1)

    elif (constraint == "Min Standard Deviation"):
        assert_smallereq(k["Solver"]["Internal"]["Current Min Standard Deviation"].getValue(), 1)

    elif (constraint == "Max Standard Deviation"):
        assert_greatereq(k["Solver"]["Internal"]["Current Max Standard Deviation"].getValue(), 1)

    elif (constraint == "Min Value"):
        print("TODO: Min Val")

    else:
        print("Termination Criterion not recognized!")


#################################################
# Main (called from run_test.sh with arg)
#################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cmaes_termination', description='Check Termination Criterion.')
    parser.add_argument('--constraint', help='Name of Constraint', action='store', required = True)
    args = parser.parse_args()

    run_ccmaes(args.constraint)
