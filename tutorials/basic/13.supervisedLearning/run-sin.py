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

e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
 
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "Input"
e["Variables"][0]["Training Data"] = tInputSet.tolist()
e["Variables"][0]["Validation Data"] = vInputSet.tolist()

e["Variables"][1]["Name"] = "Sin(X)"
e["Variables"][1]["Type"] = "Output"
e["Variables"][1]["Training Data"] = tOutputSet.tolist()
e["Variables"][1]["Validation Data"] = vOutputSet.tolist()

e["Solver"]["Type"] = "Deep Supervisor / Train"
e["Solver"]["Optimizer"] = "CMAES"

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 10
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 10
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][3]["Activation Function"] = "Identity"

e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = False

k = korali.Engine()
k.run(e)

