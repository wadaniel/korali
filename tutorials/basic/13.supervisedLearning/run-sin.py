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

testInputSet  = np.random.uniform(0,2*np.pi,500)

eTrain = korali.Experiment()

eTrain["Problem"]["Type"] = "Supervised Learning"
 
eTrain["Variables"][0]["Name"] = "X"
eTrain["Variables"][0]["Type"] = "Input"
eTrain["Variables"][0]["Training Data"] = trainingInputSet.tolist()
eTrain["Variables"][0]["Validation Data"] = validationInputSet.tolist()
eTrain["Variables"][0]["Test Data"] = testInputSet.tolist()

eTrain["Variables"][1]["Name"] = "Sin(X)"
eTrain["Variables"][1]["Type"] = "Output"
eTrain["Variables"][1]["Training Data"] = trainingOutputSet.tolist()
eTrain["Variables"][1]["Validation Data"] = validationOutputSet.tolist()

eTrain["Solver"]["Type"] = "Deep Supervisor / Train"
eTrain["Solver"]["Optimizer"] = "CMAES"

eTrain["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
eTrain["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 1
eTrain["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

eTrain["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
eTrain["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 5
eTrain["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

eTrain["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Dense"
eTrain["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 5
eTrain["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Tanh"

eTrain["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Output"
eTrain["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
eTrain["Solver"]["Neural Network"]["Layers"][3]["Activation Function"] = "Identity"

eTrain["Random Seed"] = 0xC0FFEE
k.run(eTrain)

eTest = korali.Experiment()

eTest["Problem"]["Type"] = "Supervised Learning"

eTest["Variables"][0]["Name"] = "X"
eTest["Variables"][0]["Type"] = "Input"
eTest["Variables"][0]["Test Data"] = testInputSet.tolist()

eTest["Variables"][1]["Name"] = "Sin(X)"
eTest["Variables"][1]["Type"] = "Output"

eTest["Solver"]["Type"] = "Deep Supervisor / Test"
eTest["Solver"]["Termination Criteria"]["Max Generations"] = 1
eTest["Solver"]["Neural Network"].set( eTrain["Solver"]["Neural Network"].get() )

k.run(eTest)

testOutputSet = np.sin(testInputSet).tolist()
inferredOutputSet = eTest["Results"]["Inferred Results"]

#print(testOutputSet)
#print(inferredOutputSet)

plt.plot(testInputSet,testOutputSet, "o")
plt.plot(testInputSet,inferredOutputSet, "x")
plt.show()
