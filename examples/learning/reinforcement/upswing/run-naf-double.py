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
e["Problem"]["Training Reward Threshold"] = 750
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 10

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

### Configuring NAF hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Target Learning Rate"] = 0.8
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Discount Factor"] = 1.0
e["Solver"]["Learning Rate"] = 0.000001
e["Solver"]["Mini Batch Size"] = 32

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

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

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 900
#e["Solver"]["Termination Criteria"]["Max Generations"] = 30

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
