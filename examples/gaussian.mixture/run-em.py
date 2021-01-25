#!/usr/bin/env python3

# In this example...
import numpy as np
import sys
sys.path.append('./_model')

from gaussian_mixture import gm

mean = np.array( [  [ 0,0 ], [ 5,5 ], [5,-5], [-5,-5], [-5,5 ], [0,-10 ] ] )
N = mean.shape[0]
Nd = mean.shape[1]
covariance = np.zeros((N,Nd,Nd))

covariance[0] = [ [1,0],[0,1] ]
covariance[1] = [ [4,1],[1,4] ]
covariance[2] = [ [3, 1],[ 1,2] ]
covariance[3] = [ [3,-1],[-1,2] ]
covariance[4] = [ [4,-1],[-1,4] ]
covariance[5] = [ [10,0],[0,0.2] ]

weights = np.array( [1,1,1,1,1,1] )

g = gm(mean,covariance,weights)
data, _ = g.rvs(1000)
data = data.tolist()


# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Gaussian Mixture"
e["Problem"]["Number Of Distributions"] = 6
e["Problem"]["Data"] = data

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "EM"
e["Solver"]["Termination Criteria"]["Max Generations"] = 10000
e["Solver"]["Termination Criteria"]["Min Loglikelihood Difference"] = 1e-10
e["Solver"]["Termination Criteria"]["Min Hyperparameters Difference"] = 1e-10

# Configuring output settings
e["File Output"]["Frequency"] = 1
e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
