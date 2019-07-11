#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the MCMC method.

# Importing computational model
import sys
sys.path.append('./model')
from directModel import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Setting Model
k.setModel(evaluateModel)

# Selecting problem and solver types.
k["Problem"] = "Direct Evaluation"
k["Solver"]  = "MCMC"

# Defining problem's variables and their prior distribution
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["MCMC"]["Initial Mean"] = 0.0
k["Variables"][0]["MCMC"]["Standard Deviation"] = 1.0

# Configuring the MCMC sampler parameters
k["MCMC"]["Burn In"] = 500
k["MCMC"]["Max Chain Length"] = 5000
k["MCMC"]["Use Adaptive Sampling"] = True

# Reducint Output
k["Console Output Frequency"] = 500
k["File Output Frequency"] = 500

# Setting output directory
k["Result Directory"] = "_a2_sampling_mcmc_result"

# Running Korali
k.run()
