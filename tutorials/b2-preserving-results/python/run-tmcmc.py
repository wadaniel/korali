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
sys.path.append('model')
from model import *

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
def storeResults(d):
 # Gathering sample variable values
 ph = d.getVariable(0)
 temp = d.getVariable(1)
 sampleId = d.getSampleId()
 
 # Gathering QOIs
 b = getBananaCount(ph, temp)
 o = getOrangeCount(ph, temp)
 a = getAppleCount(ph, temp)
 
 # Calculating objective function
 x = getHarvestQuality(b, o, a)
 
 # Storing QOIS
 samplePHs[sampleId] = ph
 sampleTemps[sampleId] = temp
 bananaCounts[sampleId] = b
 orangeCounts[sampleId] = o
 appleCounts[sampleId] = a
 harvestQualities[sampleId] = x
 
 # Returning evaluation to Korali
 d.addResult(x)

# Starting Korali's Engine
import korali
k = korali.initialize()

# Selecting problem and solver types.
k["Problem"]["Type"] = "Sampling"

# Defining the problem's variables and their CMA-ES bounds.
k["Variables"][0]["Name"] = "Soil pH"
k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][0]["Prior Distribution"]["Minimum"] = 3.0
k["Variables"][0]["Prior Distribution"]["Maximum"] = 9.0

k["Variables"][1]["Name"] = "Average Temperature (C)"
k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform"
k["Variables"][1]["Prior Distribution"]["Minimum"] = +10.0
k["Variables"][1]["Prior Distribution"]["Maximum"] = +50.0

# Configuring the TMCMC sampler parameters
k["Solver"]["Type"] = "TMCMC"
k["Solver"]["Population Size"] = 5000
k["Solver"]["Target Coefficient of Variation"] = 0.2

# Setting Model
k.setModel(storeResults)

# Running Korali
k.run()

# Saving quantities of interest
print('Saving quantities of interest in myResults.txt...')
configFile = open("myResults.txt","w")

for i in range(len(samplePHs)):
 configFile.write(str(samplePHs[i]) + ', ' + str(sampleTemps[i]) + ', ' + str(orangeCounts[i]) + ', ' + str(bananaCounts[i]) + ', ' + str(appleCounts[i]) + ', ' + str(harvestQualities[i]) + '\n') 
 
