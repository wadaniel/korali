#!/bin/bash
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# TMCMC run method
#################################################

def run_tmcmc_with_termination_criterion(criterion, value):

    print("[Korali] Prepare DEA run with Termination Criteria "\
            "'{0}'".format(criterion))

    k = korali.initialize()

    k.setDirectModel(evaluateModel)
    
    k["Problem"]["Type"] = "Sampling"

    k["Problem"]["Variables"][0]["Name"] = "X"
    k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
    k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
    k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

    k["Solver"]["Type"] = "TMCMC"
    k["Solver"]["Population Size"] = 5000
    k["Solver"]["Covariance Scaling"] = 0.001
    k["Solver"]["Termination Criteria"][criterion] = value

    k["General"]["Results Output"]["Frequency"] = 1000

    k.run()

    if (criterion == "Max Generations"):
        assert_value(k["General"]["Current Generation"].getValue(), value)
        
    elif (criterion == "Max Model Evaluations"):
        assert_greatereq(k["General"]["Function Evaluation Count"].getValue(), value)
    
    elif (criterion == "Target Annealing Exponent"):
        assert_greatereq(k["Solver"]["Internal"]["Annealing Exponent"].getValue(), value)
    
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
    
    run_tmcmc_with_termination_criterion(args.criterion, args.value)



 
