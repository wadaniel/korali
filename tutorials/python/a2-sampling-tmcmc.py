#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the TMCMC method.

# Importing computational model
import sys
sys.path.append('./model')
from directModel import *

# Starting Korali's Engine
import korali
k = korali.initialize()

# Configuring problem and likelihood model
k["Problem"]["Type"] = "Bayesian Inference"
k["Problem"]["Likelihood"]["Model"] = "Custom"
k.setLikelihood( evaluateModel )

# Defining problem's variables and their prior distribution
k["Variables"][0]["Name"] = "X"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000

# Setting output directory
k["General"]["Results Output"]["Path"] = "_a2_sampling_tmcmc_result"
k["General"]["Console Output"]["Verbosity"] = "Detailed"
k["General"]["Random Seed"] = 1618

# Running Korali
k.run()
