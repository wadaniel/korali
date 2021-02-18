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
e["Variables"][1]["Name"] = "Angle 1"
e["Variables"][2]["Name"] = "Angle 2"
e["Variables"][3]["Name"] = "Car Velocity"
e["Variables"][4]["Name"] = "Angular Velocity 1"
e["Variables"][5]["Name"] = "Angular Velocity 2"
e["Variables"][6]["Name"] = "Height Proxy"

e["Variables"][7]["Name"] = "Force"
e["Variables"][7]["Type"] = "Action"
e["Variables"][7]["Lower Bound"] = -20.0
e["Variables"][7]["Upper Bound"] = +20.0
e["Variables"][7]["Initial Exploration Noise"] = 2.0

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 900

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Policy Distribution"] = "Normal"
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch Size"] = 256

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

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

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
