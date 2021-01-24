#!/usr/bin/env python3

# In this example...
import numpy as np
import sys
sys.path.append('./_model')

from gaussian_mixture import gm

mean = np.array( [  [ 0,0 ], [ 4,4 ], [-4,4], [4,-4], [-4,-4 ] ] )
N = mean.shape[0]
Nd = mean.shape[1]
covariance = np.zeros((N,Nd,Nd))

covariance[0] = [ [1,0],[0,1] ]
covariance[1] = [ [1,0],[0,1] ]
covariance[2] = [ [4,1],[1,4] ]
covariance[3] = [ [4,1],[1,4] ]
covariance[4] = [ [3,-1],[-1,2] ]

weights = np.array( [1,2,1,1,2] )

g = gm(mean,covariance,weights)
data, _ = g.rvs(1000)
data = data.tolist()


# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Gaussian Mixture"
e["Problem"]["Number Of Distributions"] = 5
e["Problem"]["Data"] = data

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "EM"
e["Solver"]["Termination Criteria"]["Max Generations"] = 10000
e["Solver"]["Termination Criteria"]["Min Loglikelihood Difference"] = 1e-8
e["Solver"]["Termination Criteria"]["Min Hyperparameters Difference"] = 1e-8

# Configuring output settings
e["File Output"]["Frequency"] = 1
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
