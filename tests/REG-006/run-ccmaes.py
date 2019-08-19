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
    k["Problem"]["Variables"][0]["Lower Bound"] = -10.0
    k["Problem"]["Variables"][0]["Upper Bound"] = +10.0
 
    k["Problem"]["Variables"][1]["Name"] = "Y"
    k["Problem"]["Variables"][1]["Lower Bound"] = -10.0
    k["Problem"]["Variables"][1]["Upper Bound"] = +10.0


    k["Solver"]["Type"] = "CMAES"
    k["Solver"]["Population Size"] = 8
    k["Solver"]["Viability Population Size"] = 2
    k["Solver"]["Termination Criteria"]["Max Generations"] = 100
    k["Solver"]["Is Sigma Bounded"] = 1

    k["General"]["Console Output"]["Verbosity"] = "Detailed"
    k["General"]["Results Output"]["Frequency"] = 1000
    k["General"]["Random Seed"] = 1337


    if (constraint == "None"):
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-6*1e-17)

    elif (constraint == "Inactive"):
        k.addConstraint(inactive1)
        k.addConstraint(inactive2)
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-1.8*1e-20)

    elif (constraint == "Active at Max 1"):
        k.addConstraint(activeMax1)
        k.addConstraint(activeMax2)
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-4.826824e+00)

    elif (constraint == "Active at Max 2"):
        k.addConstraint(activeMax1)
        k.addConstraint(activeMax2)
        k.addConstraint(activeMax3)
        k.addConstraint(activeMax4)
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-9.653645e+00)


    elif (constraint == "Inactive at Max 1"):
        k.addConstraint(inactiveMax1)
        k.addConstraint(inactiveMax2)
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-2.19963e-19)
 
    elif (constraint == "Inactive at Max 2"):
        k.addConstraint(inactiveMax1)
        k.addConstraint(inactiveMax2)
        k.addConstraint(inactiveMax3)
        k.addConstraint(inactiveMax4)
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-4.626392e-17)
 
    elif (constraint == "Mixed"):
        k.addConstraint(activeMax1)
        k.addConstraint(activeMax2)
        k.addConstraint(activeMax3)
        k.addConstraint(activeMax4)
        k.addConstraint(inactiveMax1)
        k.addConstraint(inactiveMax2)
        k.addConstraint(inactiveMax3)
        k.addConstraint(inactiveMax4)
        k.run()
        assert_greatereq(k["Solver"]["Internal"]["Best Ever Value"].getValue(),-7.895685e+01)
   
    elif (constraint == "Stress"):
        k.addConstraint(activeMax1)
        k.addConstraint(activeMax2)
        k.addConstraint(activeMax3)
        k.addConstraint(activeMax4)
        k.addConstraint(inactiveMax1)
        k.addConstraint(inactiveMax2)
        k.addConstraint(inactiveMax3)
        k.addConstraint(inactiveMax4)
        k.addConstraint(stress1)
        k.addConstraint(stress2)
        k.addConstraint(stress3)
        k.addConstraint(stress4)
        k.addConstraint(stress5)
        k.addConstraint(stress6)
        k.addConstraint(stress7)
        k.addConstraint(stress8)
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
