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
e["Problem"]["Computational Model"] = lambda sampleData: modelWithGradientsAndHessians(
    sampleData, getReferencePoints())

# Configuring HMC parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Version"] = "Riemannian"
e["Solver"]["Inverse Regularization Parameter"] = 0.1
e["Solver"]["Max Num Fixed Point Iteration"] = 5
e["Solver"]["Burn In"] = 500
e["Solver"]["Use NUTS"] = False
e["Solver"]["Use Diagonal Metric"] = True
e["Solver"]["Max Depth"] = 10
e["Solver"]["Num Integration Steps"] = 20
#e["Solver"]["Target Integration Time"] = 0.5
e["Solver"]["Step Size"] = 1.0
e["Solver"]["Use Adaptive Step Size"] = True
#e["Solver"]["Desired Average Acceptance Rate"] = 0.75
e["Solver"]["Integrator Verbosity"] = False
e["Solver"]["Hamiltonian Verbosity"] = False
e["Solver"]["Termination Criteria"]["Max Samples"] = 10000

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
e["Distributions"][2]["Maximum"] = +5.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Mean"] = 2.50
e["Variables"][0]["Initial Standard Deviation"] = 1.0

e["Variables"][1]["Name"] = "b"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][1]["Initial Mean"] = 2.50
e["Variables"][1]["Initial Standard Deviation"] = 1.0

e["Variables"][2]["Name"] = "[Sigma]"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"
e["Variables"][2]["Initial Mean"] = 2.50
e["Variables"][2]["Initial Standard Deviation"] = 1.0

# Configuring output settings
e["File Output"]["Frequency"] = 50000
e["Console Output"]["Frequency"] = 500

e["File Output"]["Path"] = '_korali_result_hmc'
e["Console Output"]["Verbosity"] = "Normal"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
