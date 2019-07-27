#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the TMCMC method.

# Importing computational model
import sys
import korali

thetaFiles = [ 'results/individual/001/s00009.json', \
               'results/individual/002/s00011.json', \
               'results/individual/003/s00012.json', \
               'results/individual/004/s00013.json', \
               'results/individual/005/s00010.json' \
             ]

N = len(thetaFiles)

k=[]
for i in range(N):
  k.append( korali.initialize() )
  k[i].loadState(thetaFiles[i])


# Creating hierarchical Bayesian problem from previous two problems
kH = korali.initialize()

kH["Problem"]["Type"]  = "Hierarchical Bayesian"
kH["Problem"]["Model"] = "Sample Psi"
for i in range(N):
  kH["Problem"]["Sub-Problems"][i] = k[i].getResults();

# Add probability of theta given psi
kH["Problem"]["Conditional Priors"][0]["Type"] = "Gaussian"
kH["Problem"]["Conditional Priors"][0]["Mean"] = "Psi 1"
kH["Problem"]["Conditional Priors"][0]["Standard Deviation"] = "Psi 2"

kH["Problem"]["Conditional Priors"][1]["Type"] = "Uniform"
kH["Problem"]["Conditional Priors"][1]["Minimum"] = 1.0
kH["Problem"]["Conditional Priors"][1]["Maximum"] = "Psi 3"

kH["Variables"][0]["Name"] = "Psi 1"
kH["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][0]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][0]["Prior Distribution"]["Maximum"] = +10.0

kH["Variables"][1]["Name"] = "Psi 2"
kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][1]["Prior Distribution"]["Minimum"] =   0.0
kH["Variables"][1]["Prior Distribution"]["Maximum"] = +10.0

kH["Variables"][1]["Name"] = "Psi 3"
kH["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
kH["Variables"][1]["Prior Distribution"]["Minimum"] = -10.0
kH["Variables"][1]["Prior Distribution"]["Maximum"] = +10.0

kH["Solver"]["Type"] = "TMCMC"
kH["Solver"]["Population Size"] = 5000

kH["General"]["Max Generations"] = 100
kH["General"]["Results Output"]["Frequency"] = 20
kH["General"]["Console Output"]["Frequency"] = 20

kH.dry()
