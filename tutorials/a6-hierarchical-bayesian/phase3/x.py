#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali
sys.path.append('../setup/model')
from model import *

# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"]  = "Hierarchical Bayesian (Theta)"

kH["Problem"]["Sub Problem"] = '../setup/results_phase_1/000/final.json'
kH["Problem"]["Psi Problem"] = '../setup/results_phase_2/final.json'

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000
kH["Solver"]["Termination Criteria"]["Max Generations"] = 10
# kH["Solver"]["Burn In Steps"] = [ 3 ];
kH["Solver"]["Target Coefficient Of Variation"] = 0.6

kH["General"]["Console Output"]["Verbosity"] = "Detailed"

dataPath    = "../setup/data/"
x = getReferencePoints(dataPath,0);
kH.setReferenceModel( lambda d: logistic( x, d) )

kH.run()
