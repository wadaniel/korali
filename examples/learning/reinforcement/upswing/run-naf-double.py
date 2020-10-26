#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from double_env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 10 

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Angle 1"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Angle 2"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Car Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Angular Velocity 1"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Angular Velocity 2"
e["Variables"][5]["Type"] = "State"

e["Variables"][6]["Name"] = "Height Proxy"
e["Variables"][6]["Type"] = "State"

e["Variables"][7]["Name"] = "Force"
e["Variables"][7]["Type"] = "Action"
e["Variables"][7]["Lower Bound"] = -20.0
e["Variables"][7]["Upper Bound"] = +20.0
e["Variables"][7]["Exploration Sigma"] = 5.0

### Configuring NAF hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Target Learning Rate"] = 0.7
e["Solver"]["Optimization Steps Per Update"] = 1
e["Solver"]["Experiences Between Agent Trainings"] = 1
e["Solver"]["Experiences Between Target Network Updates"] = 1

e["Solver"]["Random Action Probability"]["Initial Value"] = 0.5
e["Solver"]["Random Action Probability"]["Target Value"] = 0.01
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.03

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

## Defining Q-Network

e["Solver"]["Critic"]["Discount Factor"] = 1.0
e["Solver"]["Critic"]["Learning Rate"] = 0.0001
e["Solver"]["Critic"]["Mini Batch Size"] = 32
e["Solver"]["Critic"]["Normalization Steps"] = 32

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 750
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 900
e["Solver"]["Termination Criteria"]["Max Generations"] = 30


### Setting file output configuration

e["File Output"]["Frequency"] = 1000

### Running Experiment

k.run(e)
