#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


######## Defining Auxiliar Functions

def gaussian(x, alpha, r):
    return 1./(math.sqrt(alpha**math.pi))*np.exp(-alpha*np.power((x - r), 2.))

######## Defining Problem's Constants

N = 10 # Max stages
initialX = 1 # Initial state

####### Defining Problem's environment

def env(s):

 # Reading initial state
 x = s["State"][0]

 for i in range(N):
  
  # Reading action
  y = s["Action"][0]
  
  print('Running Environment')
  print(' + State:  ' + str(x))
  print(' + Action: ' + str(y))
  
  # If y less than zero, no reward 
  if (y < 0): 
   s["Reward"] = 0
   return
   
  # If y if higher than x, then return zero 
  if (y > x): 
   s["Reward"] = 0
   return
   
  # Calcuating reward
  reward = (y*y)/3
     
  s["Reward"] = reward
  print(' + Reward: ' + str(reward))
  
  # Storing new state
  x = x - abs(y)
  
  # If state < 0 then we are finished
  if (x <= 0): return
  
  s["State"] = [ x ]
  
  # Reporting back and getting new action
  s.update()
 
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env
e["Problem"]["Initial States"] = [ [ initialX ] for i in range(1000000) ]

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = 1.0

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "DQN"
e["Solver"]["Replay Start Size"] = 50
e["Solver"]["Replay Memory Size"] = 1000
e["Solver"]["Agent History Size"] = 2
e["Solver"]["Mini Batch Size"] = 4
e["Solver"]["Optimization Steps Per Update"] = 10
e["Solver"]["Discount Factor"] = 0.99

### Defining the shape of the neural network

e["Solver"]["Action Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Action Optimizer"]["Termination Criteria"]["Max Gradient Norm"] = 1000000000
#e["Solver"]["Action Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = 0.0000000010

e["Solver"]["Weight Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Weight Optimizer"]["Termination Criteria"]["Max Gradient Norm"] = 1000000000
e["Solver"]["Weight Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = 0.0000000000001

e["Solver"]["Neural Network"]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 20
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"] = "Identity" 

e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = False

### Training the neural network

k.run(e)

