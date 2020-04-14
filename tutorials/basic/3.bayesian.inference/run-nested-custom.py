#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior distribution
# from the gaussian likelihood function with uniform prior.

# Importing the computational model
import sys
sys.path.append('./model')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = lambda sampleData: customModel(sampleData)

# Configuring Nested Sampling parameters
e["Solver"]["Type"] = "Nested"
e["Solver"]["Number Live Points"] = 1500
e["Solver"]["Batch Size"] = 1
e["Solver"]["Covariance Scaling"] = 2.0
e["Solver"]["Add Live Points"] = True
e["Solver"]["Resampling Method"] = "Box"

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -5.0
e["Distributions"][0]["Maximum"] = +5.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["File Output"]["Frequency"] = 0
e["Console Output"]["Frequency"] = 500
e["Solver"]["Termination Criteria"]["Max Generations"] = 6500
e["Solver"]["Termination Criteria"]["Max Gain Factor"] = 1e-9

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
