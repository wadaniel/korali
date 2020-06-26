#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

######## Defining Problem's Constants

N = 3 # Number of stages
initialX = 1.0 # Initial value of X
alpha = 0.75 # Alpha
beta = 0.50 # Beta

######## Defining Problem's Formulae

# Reward Function g(y)
def g(y):
 return np.cos(1 + 23 * y)

# Reward Function h(x-y)
def h(v):
 return np.sin(10 * v)

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
  
  # Calcuating reward
  reward = g(y) + h(x-y)
  
  # If Y is less than zero, reduce reward and finish
  if (y < 0): 
   reward = reward - 1000
   print(' + Reward: ' + str(s["Reward"]))
   return
  
  # If Y greater than current X, then this is not a feasible policy, with discount for the extra expenditure
  if (y > x): 
   reward = reward + (x - y)*100.0
   print(' + Reward: ' + str(s["Reward"]))
   return  
     
  s["Reward"] = reward
  print(' + Reward: ' + str(s["Reward"]))
  
  # Storing new state
  x = alpha*y + beta*(x-y)
  s["State"] = [ x ]
  
  # Reporting back and getting new action
  s.update()
 
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env
e["Problem"]["Initial States"] = [ [ initialX ] for i in range(3000) ]

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = 1.0

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "DQN"

### Defining the shape of the neural network

e["Solver"]["Action Optimizer"]["Type"] = "Optimizer/Adam"
#e["Solver"]["Action Optimizer"]["Termination Criteria"]["Max Infeasible Resamplings"] = 10000;
#e["Solver"]["Action Optimizer"]["Termination Criteria"]["Max Generations"] = 1000;

e["Solver"]["Weight Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Neural Network"]["Batch Normalization"]["Enabled"] = False

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

