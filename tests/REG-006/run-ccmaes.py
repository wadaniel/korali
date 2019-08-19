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

    k["General"]["Console Output"]["Verbosity"] = "Detailed"
    #k["General"]["Results Output"]["Frequency"] = 1000
    k["General"]["Random Seed"] = 1337


    if (constraint == "Inactive"):
        k.addConstraint(inactive)

    else:
        print("Constraint not recognized!")

    k.run()

#################################################
# Main (called from run_test.sh with arg)
#################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cmaes_termination', description='Check Termination Criterion.')
    parser.add_argument('--constraint', help='Name of Constraint', action='store', required = True)
    args = parser.parse_args()

    run_ccmaes(args.constraint)
