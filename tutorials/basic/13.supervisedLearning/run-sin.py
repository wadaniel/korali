#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

scaling = 5.0

import korali
k = korali.Engine()

# Defining Training Sets
np.random.seed(0xC0FFEE)
trainingInputSet  = np.random.uniform(0,2*np.pi,200)
validationInputSet  = np.random.uniform(0,2*np.pi,200)

trainingOutputSet = np.sin(trainingInputSet)*scaling
validationOutputSet = np.sin(validationInputSet)*scaling

e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Learning"
 
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "Input"
e["Variables"][0]["Training Data"] = trainingInputSet.tolist()
e["Variables"][0]["Validation Data"] = validationInputSet.tolist()

e["Variables"][1]["Name"] = "Sin(X)"
e["Variables"][1]["Type"] = "Output"
e["Variables"][1]["Training Data"] = trainingOutputSet.tolist()
e["Variables"][1]["Validation Data"] = validationOutputSet.tolist()

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Deep Learning"
e["Solver"]["Termination Criteria"]["Max Inactive Steps"] = 50

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Optimizer"]["Type"] = "CMAES"

e["Solver"]["Neural Network"]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 10
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Identity"

e["Random Seed"] = 0xC0FFEE

### Training the neural network

k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = np.random.uniform(0,2*np.pi,200)
testInputSet = [ [ x ] for x in testInputSet.tolist() ]

testInferredSet = e.test(testInputSet)
testOutputSet = np.sin(testInputSet)*scaling

### Plotting Results

plt.plot(testInputSet,testOutputSet, "o")
plt.plot(testInputSet,testInferredSet, "x")
plt.show()
