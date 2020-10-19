#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = [ [ -100.0 ], [ -90.0 ], [ -80.0 ], [ -70.0 ], [ -60.0 ], [ -50.0 ], [ -40.0 ], [ -30.0 ], [ -20.0 ], [ -10.0 ], [0.0], [  10.0 ], [ 20.0 ], [ 30.0 ], [  40.0 ], [  50.0 ], [  60.0 ], [ 70.0 ], [ 80.0 ], [ 90.0 ], [ 100.0 ]  ]
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Angle 1"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Angular Velocity 1"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Angle 2"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Angular Velocity 2"
e["Variables"][5]["Type"] = "State"

e["Variables"][6]["Name"] = "Force"
e["Variables"][6]["Type"] = "Action"

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "Agent / Discrete / DQN"
e["Solver"]["Experiences Between Updates"] = 100
e["Solver"]["Optimization Steps Per Update"] = 1

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 5000
e["Solver"]["Experience Replay"]["Maximum Size"] = 150000

### Defining probability of taking a random action (epsilon)

e["Solver"]["Random Action Probability"]["Initial Value"] = 1.0
e["Solver"]["Random Action Probability"]["Target Value"] = 0.05
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.10

## Defining Q-Critic and Action-selection (policy) optimizers

e["Solver"]["Policy"]["Mini Batch Size"] = 32
e["Solver"]["Policy"]["Learning Rate"] = 0.1
e["Solver"]["Critic"]["Discount Factor"] = 1.0

### Defining the shape of the neural network

e["Solver"]["Policy"]["Normalization Steps"] = 32

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = True 

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 500
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 500
e["Solver"]["Termination Criteria"]["Max Generations"] = 15000

### Setting file output configuration

e["File Output"]["Frequency"] = 1000

### Running Experiment

k.run(e)
