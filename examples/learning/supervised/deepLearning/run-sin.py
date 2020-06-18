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

trainingInputSet = np.random.uniform(0, 2 * np.pi, 500)
trainingSolutionSet = np.sin(trainingInputSet) * scaling

validationInputSet = np.random.uniform(0, 2 * np.pi, 500)
validationSolutionSet = np.sin(validationInputSet) * scaling

trainingInputSet = [ [ i ] for i in trainingInputSet.tolist() ]
trainingSolutionSet = [ [ i ] for i in trainingSolutionSet.tolist() ]

validationInputSet = [ [ i ] for i in validationInputSet.tolist() ]
validationSolutionSet = [ [ i ] for i in validationSolutionSet.tolist() ]

e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Supervised Learning"

e["Problem"]["Training"]["Input"] = trainingInputSet
e["Problem"]["Training"]["Solution"] = trainingSolutionSet

e["Problem"]["Validation"]["Input"] = validationInputSet
e["Problem"]["Validation"]["Solution"] = validationSolutionSet

e["Variables"][0]["Name"] = "tt"

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Deep Learner"

e["Solver"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Steps Per Generation"] = 100

e["Solver"]["Termination Criteria"]["Max Inactive Steps"] = 10
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 5
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Identity"

e["Console Output"]["Frequency"] = 10
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = np.random.uniform(0, 2 * np.pi, 100)
testInputSet = [[x] for x in testInputSet.tolist()]

testInferredSet = e.evaluate(testInputSet)
testOutputSet = np.sin(testInputSet) * scaling

### Plotting Results

plt.plot(testInputSet, testOutputSet, "o")
plt.plot(testInputSet, testInferredSet, "x")
plt.show()
