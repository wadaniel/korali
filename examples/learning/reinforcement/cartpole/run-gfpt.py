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
e["Problem"]["Actions Between Policy Updates"] = 1
e["Problem"]["Training Reward Threshold"] = 450
e["Problem"]["Policy Testing Episodes"] = 10

### Defining state variables

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][3]["Name"] = "Pole Angular Velocity"

### Defining action variables 

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

### Defining the configuration of replay memory

e["Solver"]["Mini Batch Strategy"] = "Uniform"
e["Solver"]["Experience Replay"]["Start Size"] =   2000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000
e["Solver"]["Experience Replay"]["Serialization Frequency"] = 10

## Defining Critic Configuration

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 0.01
e["Solver"]["Critic"]["Mini Batch Size"] = 32 
  
## Defining Policy Configuration

e["Solver"]["Policy"]["Learning Rate"] = 0.01
e["Solver"]["Policy"]["Mini Batch Size"] = 32
e["Solver"]["Policy"]["Target Accuracy"] = 0.00001

### Configuring the neural network's hidden layers

e["Solver"]["Engine"] = "OneDNN"

e["Solver"]["Hidden Layers][0]["Type"] = "Layer/Linear"
e["Solver"]["Hidden Layers][0]["Node Count"] = 32

e["Solver"]["Hidden Layers][1]["Type"] = "Layer/Activation"
e["Solver"]["Hidden Layers][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Hidden Layers][2]["Type"] = "Layer/Linear"
e["Solver"]["Hidden Layers][2]["Node Count"] = 32

e["Solver"]["Hidden Layers][3]["Type"] = "Layer/Activation"
e["Solver"]["Hidden Layers][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 450

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 10
 
### Running Training Experiment

#k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 5
k.run(e)

### Running Testing Experiment

e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = [ 0, 1, 2 ]

k.run(e)
