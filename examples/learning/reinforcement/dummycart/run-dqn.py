#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cart problem's configuration

actions = [ [i] for i in range(-10,11) ]

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = actions
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Force"
e["Variables"][1]["Type"] = "Action"

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "Agent / Discrete / DQN"
e["Solver"]["Optimization Steps Per Update"] = 1
e["Solver"]["Experiences Between Agent Trainings"] = 1

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 500
e["Solver"]["Experience Replay"]["Maximum Size"] = 150000

### Defining probability of taking a random action (epsilon)

e["Solver"]["Random Action Probability"]["Initial Value"] = 1.0
e["Solver"]["Random Action Probability"]["Target Value"] = 0.05
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.10

## Defining Q-Critic and Action-selection (policy) optimizers

e["Solver"]["Critic"]["Mini Batch Size"] = 32
e["Solver"]["Critic"]["Learning Rate"] = 0.01
e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Target Update Delay"] = 5

### Defining the shape of the neural network

e["Solver"]["Critic"]["Normalization Steps"] = 16

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = True 

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 95
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 95

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
