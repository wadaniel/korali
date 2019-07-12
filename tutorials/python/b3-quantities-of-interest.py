#!/usr/bin/env python3

## In this example, we demonstrate how to use Korali to sample
## a computational model, while saving important quantities of
## interest derived from the model

# Our computational model calculates the harvest yield of 
# bananas, apples, and oranges from a farming field with
# respect to the soil pH and average temperatures.

# The main quantity of interest is the 'harvest quality',
# which measures the nutritional value of the harvest as
# a whole. The more produce, the better harvest. This is
# the value we want to sample from.

# However, we also want to store the number of individual
# fruit counts during our sampling. In this example, we
# see how to keep these values. 
 
# Importing computational model
import sys
sys.path.append('./model')
from qoi import *

# Allocating storage for Quantities of Interest
# In this case we use a dictionary that associates the
# quantities of interests to a specific sample Id, provided
# by Korali.
samplePHs = {}
sampleTemps = {}
bananaCounts = {}
orangeCounts = {}
appleCounts = {}
harvestQualities = {}

# We run the computational model for each sample, storing
# individual fruit counts, but also returning the harvest
# quality back to Korali to sample around the parameter 
# space with higher yield.
def evaluateModel(k):
 # Gathering sample variable values
 ph = k.getVariable(0)
 temp = k.getVariable(1)
 sampleId = k.getSampleId()
 x, b, o, a = estimateHarvest(ph, temp)
 
 # Saving quantities of interest
 samplePHs[sampleId] = ph
 sampleTemps[sampleId] = temp
 bananaCounts[sampleId] = b
 orangeCounts[sampleId] = o
 appleCounts[sampleId] = a
 harvestQualities[sampleId] = x
 
 # Returning evaluation to Korali
 k.addResult(x)

# Starting Korali's Engine
import korali
k = korali.initialize()

m = {}
m["hello"] = 2
# Selecting problem and solver types.
k["Problem"] = "Bayesian"
k["Solver"]  = "TMCMC"
k["Bayesian"]["Likelihood"]["Type"] = "Direct"
k.setLikelihood(evaluateModel)

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "Soil pH"
#k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Gaussian"
#k["Variables"][0]["Bayesian"]["Prior Distribution"]["Mean"] = +6.5
#k["Variables"][0]["Bayesian"]["Prior Distribution"]["Sigma"] = +6.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = 3.0
k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = 9.0

k["Variables"][1]["Name"] = "Average Temperature (C)"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = +10.0
k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +50.0

# Configuring the TMCMC sampler parameters
k["TMCMC"]["Population Size"] = 5000

# Setting output directory
k["Result Directory"] = "_b3_quantities_of_interest_result"

# Running Korali
k.run()

# Saving quantities of interest
print('Saving quantities of interest in qoi.txt...')
configFile = open("qoi.txt","w")

for i in range(len(samplePHs)):
 configFile.write(str(samplePHs[i]) + ', ' + str(sampleTemps[i]) + ', ' + str(orangeCounts[i]) + ', ' + str(bananaCounts[i]) + ', ' + str(appleCounts[i]) + ', ' + str(harvestQualities[i]) + '\n') 
 