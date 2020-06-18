#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def env(s):
 state = s["State"]
 action = s["Action"]
 
 print('Running Environment')
 print(' + State:  ' + str(state))
 print(' + Action: ' + str(action))
 
 s["Reward"] = 1.0
 
 s.update()

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env
e["Problem"]["Initial States"] = [ [ 0.2 ], [ 0.3 ], [ 0.5 ] ]

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = -100.0
e["Variables"][1]["Upper Bound"] = +100.0

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "DQN"

### Defining the shape of the neural network

e["Solver"]["Action Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Weight Optimizer"]["Type"] = "Optimizer/Adam"

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 5
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Identity"

e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

k.run(e)

