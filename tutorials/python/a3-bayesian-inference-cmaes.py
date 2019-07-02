#!/usr/bin/env python3

## In this example, we demonstrate how Korali find the variable values
## that maximize the posterior in a bayesian problem where the likelihood
## is calculated by providing reference data points and their objective values.

# Importing the computational model
import sys
sys.path.append('./model')
from posteriorModel import *
import korali

k = korali.initialize()

# Getting reference data from the model
x, y = getReferenceData()

Fx = lambda koraliData: evaluateModel(koraliData, x)
k.setModel( Fx )

# Selecting problem.
k["Problem"] = "Bayesian"

# Setting up the reference likelihood for the Bayesian Problem
k["Bayesian"]["Likelihood"]["Type"] = "Reference"
k["Bayesian"]["Likelihood"]["Reference Data"] = y

# Configuring the problem's variables and their prior distributions
k["Variables"][0]["Name"] = "a"
k["Variables"][0]["Bayesian"]["Type"] = "Computational"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0

k["Variables"][1]["Name"] = "b"
k["Variables"][1]["Bayesian"]["Type"] = "Computational"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0

k["Variables"][2]["Name"] = "Sigma"
k["Variables"][2]["Bayesian"]["Type"] = "Statistical"
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0
k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0

# Selecting solver type.
k["Solver"] = "CMA-ES"

# Configuring the CMA-ES bounds for the variables
k["Variables"][0]["CMA-ES"]["Lower Bound"] = -5.0
k["Variables"][0]["CMA-ES"]["Upper Bound"] = +5.0
k["Variables"][1]["CMA-ES"]["Lower Bound"] = -5.0
k["Variables"][1]["CMA-ES"]["Upper Bound"] = +5.0
k["Variables"][2]["CMA-ES"]["Lower Bound"] = 0.0
k["Variables"][2]["CMA-ES"]["Upper Bound"] = +5.0

# Configuring CMA-ES parameters
k["CMA-ES"]["Objective"] = "Maximize"
k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Value"] = 1000
k["CMA-ES"]["Sample Count"] = 5

# Reducing Output
k["Output Frequency"] = 100

# Setting output directory
k["Result Directory"] = "_a3_bayesian_inference_cmaes"

# Running Korali
k.run()
