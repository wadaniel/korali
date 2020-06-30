#!/usr/bin/env python3
import os
import sys
import math
import gym

######## Defining Environment Storage

cart = gym.make('CartPole-v1').unwrapped
maxSteps = 1000

####### Defining Problem's environment

def env(s):

 # Initializing environment
 s["State"] = cart.reset().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Reading action
  action = s["Action"][0] 
    
  # Performing the action
  state, reward, done, info = cart.step(action)

  # Storing Reward
  s["Reward"] = reward
   
  # Storing New State
  s["State"] = state.tolist()
  
  # Advancing step counter
  step = step + 1
  
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Angular Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Push Direction"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Values"] = [ 0.0, 1.0 ]

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "DQN"
e["Solver"]["Replay Start Size"] = 5000
e["Solver"]["Replay Memory Size"] = 100000
e["Solver"]["Agent History Size"] = 250
e["Solver"]["Mini Batch Size"] = 32
e["Solver"]["Optimization Steps Per Update"] = 10
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Epsilon"]["Initial Value"] = 0.9
e["Solver"]["Epsilon"]["Target Value"] = 0.05
e["Solver"]["Epsilon"]["Decrease Rate"] = 0.005

### Defining the shape of the neural network

e["Solver"]["Action Optimizer"]["Type"] = "Optimizer/Grid Search" 
e["Solver"]["Weight Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Weight Optimizer"]["Eta"] = 0.001

e["Solver"]["Neural Network"]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 4
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Identity" 

# Defining Termination Criteria
e["Solver"]["Termination Criteria"]["Max Generations"] = 400

### Training the neural network
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = True
e["File Output"]["Name"] = "result.json"
k.run(e)

