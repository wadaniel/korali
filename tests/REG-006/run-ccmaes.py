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

    k["Problem"]["Type"] = "Evaluation/Direct"
    k["Problem"]["Objective"] = "Maximize"
    k["Problem"]["Objective Function"] = evaluateModel

    k["Variables"][0]["Name"] = "X"
    k["Variables"][0]["Lower Bound"] = -10.0
    k["Variables"][0]["Upper Bound"] = +10.0
 
    k["Variables"][1]["Name"] = "Y"
    k["Variables"][1]["Lower Bound"] = -10.0
    k["Variables"][1]["Upper Bound"] = +10.0

    k["Solver"]["Type"] = "Optimizer/CMAES"
    k["Solver"]["Population Size"] = 8
    k["Solver"]["Viability Population Size"] = 2
    k["Solver"]["Termination Criteria"]["Max Generations"] = 100
    k["Solver"]["Is Sigma Bounded"] = 1

    k["Console Output"]["Verbosity"] = "Detailed"
    k["Results Output"]["Frequency"] = 1000
    k["Random Seed"] = 1337


    if (constraint == "None"):
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-6*1e-17)

    elif (constraint == "Inactive"):
        k["Problem"]["Constraints"][0] = inactive1
        k["Problem"]["Constraints"][1] = inactive2
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-1.8*1e-20)

    elif (constraint == "Active at Max 1"):
        k["Problem"]["Constraints"][0] = activeMax1
        k["Problem"]["Constraints"][1] = activeMax2
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-4.826824e+00)

    elif (constraint == "Active at Max 2"):
        k["Problem"]["Constraints"][0] = activeMax1
        k["Problem"]["Constraints"][1] = activeMax2
        k["Problem"]["Constraints"][2] = activeMax3
        k["Problem"]["Constraints"][3] = activeMax4
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-9.653645e+00)

    elif (constraint == "Inactive at Max 1"):
        k["Problem"]["Constraints"][0] = inactiveMax1
        k["Problem"]["Constraints"][1] = inactiveMax2
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-2.19963e-19)
 
    elif (constraint == "Inactive at Max 2"):
        k["Problem"]["Constraints"][0] = inactiveMax1
        k["Problem"]["Constraints"][1] = inactiveMax2
        k["Problem"]["Constraints"][2] = inactiveMax3
        k["Problem"]["Constraints"][3] = inactiveMax4
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-4.626392e-17)
 
    elif (constraint == "Mixed"):
        k["Problem"]["Constraints"][0] = activeMax1
        k["Problem"]["Constraints"][1] = activeMax2
        k["Problem"]["Constraints"][2] = activeMax3
        k["Problem"]["Constraints"][3] = activeMax4
        k["Problem"]["Constraints"][4] = inactiveMax1
        k["Problem"]["Constraints"][5] = inactiveMax2
        k["Problem"]["Constraints"][6] = inactiveMax3
        k["Problem"]["Constraints"][7] = inactiveMax4
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-7.895685e+01)
   
    elif (constraint == "Stress"):
        k["Problem"]["Constraints"][0] = activeMax1
        k["Problem"]["Constraints"][1] = activeMax2
        k["Problem"]["Constraints"][2] = activeMax3
        k["Problem"]["Constraints"][3] = activeMax4
        k["Problem"]["Constraints"][4] = inactiveMax1
        k["Problem"]["Constraints"][5] = inactiveMax2
        k["Problem"]["Constraints"][6] = inactiveMax3
        k["Problem"]["Constraints"][7] = inactiveMax4
        k["Problem"]["Constraints"][8] = stress1
        k["Problem"]["Constraints"][9] = stress2
        k["Problem"]["Constraints"][10] = stress3
        k["Problem"]["Constraints"][11] = stress4
        k["Problem"]["Constraints"][12] = stress5
        k["Problem"]["Constraints"][13] = stress6
        k["Problem"]["Constraints"][14] = stress7
        k["Problem"]["Constraints"][15] = stress8
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-7.895685e+01)

    else:
        print("Constraint not recognized!")
        exit(-1)


#################################################
# Main (called from run_test.sh with arg)
#################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cmaes_termination', description='Check Termination Criterion.')
    parser.add_argument('--constraint', help='Name of Constraint', action='store', required = True)
    args = parser.parse_args()

    run_ccmaes(args.constraint)
