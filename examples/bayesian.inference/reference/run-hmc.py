#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior distribution
# in a bayesian problem where the likelihood is calculated by providing
# reference data points and their objective values.

# Importing the computational model
import sys
sys.path.append('./_model')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()
e["Problem"]["Computational Model"] = lambda sampleData: modelWithGradients(sampleData, getReferencePoints())

# Configuring HMC parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Version"] = "Euclidean"
e["Solver"]["Inverse Regularization Parameter"] = 0.1
e["Solver"]["Burn In"] = 100
e["Solver"]["Use NUTS"] = True
e["Solver"]["Max Depth"] = 5
e["Solver"]["Step Size"] = 1.0
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = +5.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = +5.0

e["Distributions"][2]["Name"] = "Uniform 2"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 0.0
e["Distributions"][2]["Maximum"] = +150.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 2.50
e["Variables"][0]["Initial Standard Deviation"] = 1.0

e["Variables"][1]["Name"] = "b"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][1]["Initial Mean"] = 2.50
e["Variables"][1]["Initial Standard Deviation"] = 1.0

e["Variables"][2]["Name"] = "Dispersion"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"
e["Variables"][2]["Initial Mean"] = 2.50
e["Variables"][2]["Initial Standard Deviation"] = 1.0

# Configuring output settings
e["File Output"]["Frequency"] = 1e3
e["Console Output"]["Frequency"] = 1

e["File Output"]["Path"] = '_korali_result_hmc'
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
