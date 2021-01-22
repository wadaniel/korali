#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Mujoco inverted double pendulum configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 950
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 1

# Defining State Variables

e["Variables"][0]["Name"] = "Cart X Position"
e["Variables"][1]["Name"] = "Sin(Joint 1 Angle)"
e["Variables"][2]["Name"] = "Sin(Joint 2 Angle)"
e["Variables"][3]["Name"] = "Sin(Joint 3 Angle)"
e["Variables"][4]["Name"] = "Cos(Joint 1 Angle)"
e["Variables"][5]["Name"] = "Cos(Joint 2 Angle)"
e["Variables"][6]["Name"] = "Cos(Joint 3 Angle)"
e["Variables"][7]["Name"] = "Joint 1 Velocity"
e["Variables"][8]["Name"] = "Joint 2 Velocity"
e["Variables"][9]["Name"] = "Joint 3 Velocity"
e["Variables"][10]["Name"] = "Force Constraint"

# Defining Action Variable

e["Variables"][11]["Name"] = "Force"
e["Variables"][11]["Type"] = "Action"
e["Variables"][11]["Lower Bound"] = -1.0
e["Variables"][11]["Upper Bound"] = +1.0
e["Variables"][11]["Initial Exploration Noise"] = 0.5

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Cache Persistence"] = 200
e["Solver"]["Learning Rate"] = 0.001

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 65536

### Configuring the Remember-and-Forget Experience Replay algorithm

e["Solver"]["Experience Replay"]["REFER"]["Enabled"] = True
e["Solver"]["Experience Replay"]["REFER"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["REFER"]["Target"] = 0.2
e["Solver"]["Experience Replay"]["REFER"]["Initial Beta"] = 0.6
e["Solver"]["Experience Replay"]["REFER"]["Annealing Rate"] = 5e-7

### Configuring Mini Batch

e["Solver"]["Mini Batch Size"] = 128
e["Solver"]["Mini Batch Strategy"] = "Uniform"

## Defining Critic and Policy Configuration

e["Solver"]["Policy"]["Learning Rate Scale"] = 0.1
e["Solver"]["Policy"]["Target Accuracy"] = 0.001
e["Solver"]["Policy"]["Optimization Candidates"] = 128

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 64

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 950.0

### Setting console/file output configuration

e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 10
e["File Output"]["Path"] = "_result_gfpt"

### Running Experiment

k.run(e)

