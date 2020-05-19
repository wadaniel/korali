#!/usr/bin/env python3
import os
import sys
import korali
import numpy as np

# Defining Training Sets

tInputSet  = np.random.uniform(0,2*np.pi,100)
vInputSet  = np.random.uniform(0,2*np.pi,20)

tOutputSet = np.sin(tInputSet)
vOutputSet = np.sin(vInputSet)

trainingSet = list(zip(tInputSet.tolist(), tOutputSet.tolist()))
validationSet = list(zip(tInputSet.tolist(), tOutputSet.tolist()))

e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Training Set"] = trainingSet
e["Problem"]["Validation Set"] = validationSet
 
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "Input"

e["Variables"][1]["Name"] = "Sin(X)"
e["Variables"][1]["Type"] = "Output"

e["Solver"]["Type"] = "Deep Supervisor"
e["Solver"]["Optimizer"] = "CMAES"

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 4
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 4
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][3]["Activation Function"] = "Identity"

e["File Output"]["Enabled"] = False
e["Console Output"]["Verbosity"] = "Silent"

k = korali.Engine()
k.run(e)

