#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from single_env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Pendulum problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 750
e["Problem"]["Policy Testing Episodes"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][1]["Name"] = "Angle 1"
e["Variables"][2]["Name"] = "Car Velocity"
e["Variables"][3]["Name"] = "Angular Velocity 1"
e["Variables"][4]["Name"] = "Height Proxy"

e["Variables"][5]["Name"] = "Force"
e["Variables"][5]["Type"] = "Action"
e["Variables"][5]["Lower Bound"] = -20.0
e["Variables"][5]["Upper Bound"] = +20.0
e["Variables"][5]["Exploration Sigma"]["Initial"] = 1.00
e["Variables"][5]["Exploration Sigma"]["Final"] = 1.00
e["Variables"][5]["Exploration Sigma"]["Annealing Rate"] = 0.00

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Time Sequence Length"] = 1
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Cache Persistence"] = 10
e["Solver"]["Discount Factor"] = 0.99

### Defining the configuration of replay memory

e["Solver"]["Mini Batch Size"] = 32
e["Solver"]["Mini Batch Strategy"] = "Uniform"
e["Solver"]["Experience Replay"]["Start Size"] =   1024
e["Solver"]["Experience Replay"]["Maximum Size"] = 32768
e["Solver"]["Experience Replay"]["Serialization"]["Frequency"] = 10

## Defining Critic and Policy Configuration

e["Solver"]["Learning Rate"] = 0.01
e["Solver"]["Policy"]["Learning Rate Scale"] = 1.0
e["Solver"]["Critic"]["Advantage Function Population"] = 12
e["Solver"]["Policy"]["Target Accuracy"] = 0.001
e["Solver"]["Policy"]["Optimization Candidates"] = 12

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

e["Problem"]["Training Reward Threshold"] = 750
e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 900

### Setting file output configuration

#e["Console Output"]["Verbosity"] = "Silent"
e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
