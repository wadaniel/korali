#!/usr/bin/env python3

# In this example, we demonstrate how a Korali experiment can
# be resumed from any point (generation). This is a useful feature
# for continuing jobs after an error, or to fragment big jobs into
# smaller ones that can better fit a supercomputer queue.
#
# First, we run a simple Korali experiment.

import sys
sys.path.append('./model')
from model import *

import korali
k = korali.initialize()

k["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Custom"
k["Problem"]["Likelihood Model"] = calculateLogLikelihood

k["Solver"]["Type"]  = "Sampler/TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Termination Criteria"]["Generations Per Run"] = 2

k["Distributions"][0]["Name"] = "Uniform 0"
k["Distributions"][0]["Type"] = "Univariate/Uniform"
k["Distributions"][0]["Minimum"] = -10.0
k["Distributions"][0]["Maximum"] = +10.0

k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"] = "Uniform 0"

k["Result Path"] = '_result_run-tmcmc'
k["Random Seed"] = 0xC0FFEE

print("\n-------------------------------------------------------------")
print("Running first 2 generations...")
print("-------------------------------------------------------------\n")

k.run()

print("\n-------------------------------------------------------------")
print("Running last 2 generations...")
print("-------------------------------------------------------------\n")

k.run()

