#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the HMC method.

# Importing computational model
import sys
sys.path.append('./_model')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = lcauchy
dim = 1

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Version"] = 'Riemannian'
# e["Solver"]["Inverse Regularization Parameter"] = 0.05
# e["Solver"]["Hamiltonian Verbosity"] = True
# e["Solver"]["Integrator Verbosity"] = True
e["Solver"]["Use Adaptive Step Size"] = False
e["Solver"]["Step Size"] = 0.2
e["Solver"]["Num Integration Steps"] = 10
# e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Use NUTS"] = False
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 100000


# Defining problem's variables
# e["Variables"][0]["Name"] = "X"
# e["Variables"][0]["Initial Mean"] = 0.0
# e["Variables"][0]["Initial Standard Deviation"] = 1.0

# e["Variables"][1]["Name"] = "Y"
# e["Variables"][1]["Initial Mean"] = 0.0
# e["Variables"][1]["Initial Standard Deviation"] = 1.0

# e["Variables"][2]["Name"] = "Z"
# e["Variables"][2]["Initial Mean"] = 0.0
# e["Variables"][2]["Initial Standard Deviation"] = 1.0


# e["Solver"]["Use NUTS"] = False
# e["Solver"]["Version"] = 'Euclidean'
# e["Solver"]["Use Diagonal Metric"] = True
# e["Solver"]["Use Adaptive Step Size"] = False
# e["Solver"]["Step Size"] = 0.2
# e["Solver"]["Num Integration Steps"] = 20
# e["Solver"]["Max Depth"] = 10
# e["Solver"]["Use Riemannian Metric"] = True
# e["Solver"]["Integrator Verbosity"] = False
# e["Solver"]["Hamiltonian Verbosity"] = False

for i in range(dim):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Initial Mean"] = 1.0
    e["Variables"][i]["Initial Standard Deviation"] = 1.0

# Configuring output settings
e["File Output"]["Frequency"] = 1000
e["Console Output"]["Frequency"] = 1000
e["Console Output"]["Verbosity"] = "Detailed"

e["Random Seed"] = 0xC0FFEE

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

samples = e["Results"]["Sample Database"]
print("############################################################")
samples = np.reshape(samples, (-1, dim))
print(samples)
print(np.mean(samples, axis=0))
