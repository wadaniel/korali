#!/usr/bin/env python3
import os
import sys
sys.path.append('../../learning/reinforcement/cartpole/_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Loading previous run (if exist)

found = e.loadState('_result_gfpt/latest')

# If not found, we run first 10 generations.
if (found == False):
  e["Solver"]["Termination Criteria"]["Max Generations"] = 10
  print('------------------------------------------------------')
  print('Running first 10 generations...')
  print('------------------------------------------------------')

# If found, we continue 
if (found == True):
  print('------------------------------------------------------')
  print('Running 10 more generations...')
  print('------------------------------------------------------')
  e["Solver"]["Termination Criteria"]["Max Generations"] = 20
  
### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Actions Between Policy Updates"] = 1
e["Problem"]["Training Reward Threshold"] = 450
e["Problem"]["Policy Testing Episodes"] = 10

### Defining State variables

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][3]["Name"] = "Pole Angular Velocity"

### Defining Action variables 

e["Variables"][4]["Name"] = "Force"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -10.0
e["Variables"][4]["Upper Bound"] = +10.0
e["Variables"][4]["Exploration Sigma"] = 0.35

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Agent Count"] = 5
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

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 4500

### Setting file output configuration

e["File Output"]["Path"] = "_result_gfpt"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
 
# Configuring Korali's Engine
k.run(e)
