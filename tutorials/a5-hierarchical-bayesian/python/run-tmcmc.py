#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the TMCMC method.

# Importing computational model
import sys
sys.path.append('model')
from model import *
import korali

# Running first Bayesian problem
k1 = korali.initialize()

k1["Problem"]["Type"] = "Bayesian Inference"
k1["Problem"]["Likelihood"]["Model"] = "Custom"

k1["Variables"][0]["Name"] = "X"
k1["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k1["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k1["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

k1["Solver"]["Type"] = "TMCMC"
k1["Solver"]["Population Size"] = 5000

k1["General"]["Console Output"]["Verbosity"] = "Silent"
k1["General"]["Max Generations"] = 100

k1.setModel(myLikelihood)
k1.run()

# Running first Bayesian problem
k2 = korali.initialize()

k2["Problem"]["Type"] = "Bayesian Inference"
k2["Problem"]["Likelihood"]["Model"] = "Custom"

k2["Variables"][0]["Name"] = "Y"
k2["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k2["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k2["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

k2["Solver"]["Type"] = "TMCMC"
k2["Solver"]["Population Size"] = 5000

k2["General"]["Console Output"]["Verbosity"] = "Silent"
k2["General"]["Max Generations"] = 100

k2.setModel(myLikelihood)
k2.run()

# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"] = "Hierarchical Bayesian"
kH["Problem"]["Model"] = "Sample Psi"
kH["Problem"]["Sub-Problems"][0] = k1.getResults();
kH["Problem"]["Sub-Problems"][1] = k2.getResults();

kH["Variables"][0]["Name"] = "Psi 1"
kH["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

kH["Variables"][1]["Name"] = "Psi 2"
kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][1]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][1]["Prior Distribution"]["Maximum"] = +10.0

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000

kH["General"]["Max Generations"] = 100
kH["General"]["Results Output"]["Frequency"] = 20
kH["General"]["Console Output"]["Frequency"] = 20

kH.run()
