#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def env(s):

 # Getting first state/action
 state = s["State"][0]
 action = s["Action"][0]
  
 for i in range(10):
  
  print('Running Environment')
  print(' + State:  ' + str(state))
  print(' + Action: ' + str(action))
  
  # Calculating reward
  reward = state*action
  print(' + Reward: ' + str(reward))
  
  # Storing reward for current state/action 
  s["Reward"] = reward
  
  # Storing new state
  state = state - action
  s["State"] = [ state ]
  
  # Reporting back and getting new action
  s.update()
 
  # Reading new action
  action = s["Action"][0]

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
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = 0.1

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "DQN"

### Defining the shape of the neural network

e["Solver"]["Action Optimizer"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Weight Optimizer"]["Type"] = "Optimizer/Adam"

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 20
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "ReLU"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 20
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "ReLU"

e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][3]["Activation Function"] = "Identity" 

e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

k.run(e)

