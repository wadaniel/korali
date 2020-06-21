#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def env(s):

 # Getting first state/action
 state = s["State"][0]
 action = s["Action"][0]
 reward = 0
 
 while (state > 0.0):
  
  #print('Running Environment')
  #print(' + State:  ' + str(state))
  #print(' + Action: ' + str(action))
  
  # Calculating reward
  reward = reward + action
  #print(' + Reward: ' + str(reward))
  
  # Storing reward for current state/action 
  s["Reward"] = reward
  
  # Storing new state
  state = state - action
  s["State"] = [ state ]
  
  # Reporting back and getting new action
  s.update()
 
  # Reading new action
  action = s["Action"][0]
  
 # Lastly, add the remaining budget
 s["Reward"] = reward + state

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env
e["Problem"]["Initial States"] = [ [ 100.0 ] for i in range(500) ]

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = 10.0
e["Variables"][1]["Upper Bound"] = 100.0

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "DQN"
e["Solver"]["Agent Count"] = 50

### Defining the shape of the neural network

e["Solver"]["Action Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Action Optimizer"]["Termination Criteria"]["Max Generations"] = 1000;

e["Solver"]["Weight Optimizer"]["Type"] = "Optimizer/Adam"

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 10
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Identity" 

e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

k.run(e)

