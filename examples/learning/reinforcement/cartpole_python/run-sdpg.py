#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Omega"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Cos(Angle)"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Pole Sin(Angle)"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Force"
e["Variables"][5]["Type"] = "Action"
e["Variables"][5]["Lower Bound"] = -10.0
e["Variables"][5]["Upper Bound"] = +10.0

### Defining noise to add to the action

e["Variables"][5]["Exploration Noise"]["Enabled"] = True
e["Variables"][5]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal"
e["Variables"][5]["Exploration Noise"]["Distribution"]["Mean"] = 0.0
e["Variables"][5]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.35
e["Variables"][5]["Exploration Noise"]["Theta"] = 0.05

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent/Continuous/SDPG"
e["Solver"]["Normalization Steps"] = 32
e["Solver"]["Trajectory Size"] = 1
e["Solver"]["Optimization Steps Per Trajectory"] = 1

e["Solver"]["Random Action Probability"]["Initial Value"] = 0.5
e["Solver"]["Random Action Probability"]["Target Value"] = 0.05
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.05

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

## Defining Critic Configuration

e["Solver"]["Critic"]["Learning Rate"] = 0.001
e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Mini Batch Size"] = 64

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 5
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = True 

## Defining Policy Configuration

e["Solver"]["Policy"]["Learning Rate"] = 0.001
e["Solver"]["Policy"]["Mini Batch Size"] = 8 
e["Solver"]["Policy"]["Target Accuracy"] = 0.1

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 5
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Tanh" 

e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = [ 10.0 ]

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 400
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 450

### Setting file output configuration

e["File Output"]["Frequency"] = 10000
#e["Console Output"]["Verbosity"] = "Silent"

### Running Experiment

k.run(e)
