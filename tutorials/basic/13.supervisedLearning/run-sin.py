#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import korali
k = korali.Engine()

# Defining Training Sets
np.random.seed(0xC0FFEE)
trainingInputSet  = np.random.uniform(0,2*np.pi,500)
validationInputSet  = np.random.uniform(0,2*np.pi,500)

trainingOutputSet = np.sin(trainingInputSet)
validationOutputSet = np.sin(validationInputSet)

e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
 
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "Input"
e["Variables"][0]["Training Data"] = trainingInputSet.tolist()
e["Variables"][0]["Validation Data"] = validationInputSet.tolist()

e["Variables"][1]["Name"] = "Sin(X)"
e["Variables"][1]["Type"] = "Output"
e["Variables"][1]["Training Data"] = trainingOutputSet.tolist()
e["Variables"][1]["Validation Data"] = validationOutputSet.tolist()

e["Solver"]["Type"] = "Deep Supervisor / Train"

e["Solver"]["Neural Network"]["Optimizer"]["Type"] = "CMAES"
e["Solver"]["Neural Network"]["Optimizer"]["Population Size"] = 32

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 5
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 5
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][3]["Activation Function"] = "Identity"

e["Random Seed"] = 0xC0FFEE
k.run(e)

testInputSet = np.random.uniform(0,2*np.pi,500)
testInputSet = [ [ x ] for x in testInputSet.tolist() ]

testInferredSet = e.test(testInputSet)
testOutputSet = np.sin(testInputSet).tolist()

plt.plot(testInputSet,testOutputSet, "o")
plt.plot(testInputSet,testInferredSet, "x")
plt.show()
