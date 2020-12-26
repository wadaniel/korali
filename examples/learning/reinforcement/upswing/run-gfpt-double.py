#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from double_env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Pendulum problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 750
e["Problem"]["Policy Testing Episodes"] = 10
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][2]["Name"] = "Angle 1"
e["Variables"][3]["Name"] = "Angular Velocity 1"
e["Variables"][4]["Name"] = "Angle 2"
e["Variables"][5]["Name"] = "Angular Velocity 2"
e["Variables"][6]["Name"] = "Height Proxy"

e["Variables"][7]["Name"] = "Force"
e["Variables"][7]["Type"] = "Action"
e["Variables"][7]["Lower Bound"] = -20.0
e["Variables"][7]["Upper Bound"] = +20.0
e["Variables"][7]["Exploration Sigma"] = 1.0

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Agent Count"] = 5
e["Solver"]["Time Sequence Length"] = 4
e["Solver"]["Experiences Per Generation"] = 500
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Cache Persistence"] = 10
e["Solver"]["Discount Factor"] = 0.99

### Defining the configuration of replay memory

e["Solver"]["Mini Batch Size"] = 32
e["Solver"]["Mini Batch Strategy"] = "Uniform"
e["Solver"]["Experience Replay"]["Start Size"] =   2000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000
e["Solver"]["Experience Replay"]["Serialization Frequency"] = 10

## Defining Critic and Policy Configuration

e["Solver"]["Critic"]["Learning Rate"] = 0.01
e["Solver"]["Policy"]["Learning Rate"] = 0.01
e["Solver"]["Policy"]["Optimization Candidates"] = 32
e["Solver"]["Policy"]["Target Accuracy"] = 0.00001

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

### Defining Termination Criteria

e["Problem"]["Training Reward Threshold"] = 750
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 900

### Setting file output configuration

#e["Console Output"]["Verbosity"] = "Silent"
e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
