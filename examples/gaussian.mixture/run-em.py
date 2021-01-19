#!/usr/bin/env python3

# In this example...

# Data
data = [[1,1],[2,2],[1,2],[2,1],[4,4],[-1,-2],[-1,-1],[0,0],[1,-2]]

import numpy as np
# print(np.mean(data,axis=0))
# print('-------------------')
# print(np.cov(data,rowvar=False))
# print('-------------------')

# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Gaussian Mixture"
e["Problem"]["Number Of Distributions"] = 2
e["Problem"]["Data"] = data

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "EM"
e["Solver"]["Termination Criteria"]["Max Generations"] = 5

# Configuring output settings
e["File Output"]["Frequency"] = 1
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
