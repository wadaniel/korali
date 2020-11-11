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

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Force"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = -1.0
e["Variables"][1]["Upper Bound"] = +1.0
e["Variables"][1]["Exploration Sigma"] = 0.35

### Configuring NAF hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Target Learning Rate"] = 0.001
e["Solver"]["Optimization Steps Per Update"] = 100
e["Solver"]["Experiences Between Agent Trainings"] = 10
e["Solver"]["Covariance Scaling"] = 0.01
e["Solver"]["Mini Batch Strategy"] = "Prioritized"

e["Solver"]["Random Action Probability"]["Initial Value"] = 0.0
e["Solver"]["Random Action Probability"]["Target Value"] = 0.0
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.03

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   200
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

## Defining Q-Network

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 1e-5
e["Solver"]["Critic"]["Mini Batch Size"] = 16

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 16
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 16
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 95
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 95

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
