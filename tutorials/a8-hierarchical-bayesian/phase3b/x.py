#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali
sys.path.append('../setup/model')
from model import *

# Creating hierarchical Bayesian problem from previous two problems
e = korali.newExperiment()

k["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/Theta"
k["Problem"]["Theta Problem Path"] = '../setup/results_phase_1/000/'
k["Problem"]["Psi Problem Path"] = '../setup/results_phase_2/'

k["Solver"]["Type"] = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 1000
k["Solver"]["Termination Criteria"]["Max Generations"] = 30
k["Solver"]["Default Burn In"] = 1;
k["Solver"]["Target Coefficient Of Variation"] = 0.6

e["Verbosity"] = "Detailed"
e["Result Path"] = "../setup/results_phase_3b/"

# Starting Korali's Engine and running experiment
k = korali.initialize()
k["Conduit"]["Type"] = "External"
k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)
