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
e["Problem"]["Training Reward Threshold"] = 750
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Angle 1"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Car Velocity"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Angular Velocity 1"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "A"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "B"
e["Variables"][5]["Type"] = "State"

e["Variables"][6]["Name"] = "C"
e["Variables"][6]["Type"] = "State"

e["Variables"][7]["Name"] = "D"
e["Variables"][7]["Type"] = "State"

e["Variables"][8]["Name"] = "E"
e["Variables"][8]["Type"] = "State"

e["Variables"][9]["Name"] = "F"
e["Variables"][9]["Type"] = "State"

e["Variables"][10]["Name"] = "G"
e["Variables"][10]["Type"] = "State"

e["Variables"][11]["Name"] = "Force"
e["Variables"][11]["Type"] = "Action"
e["Variables"][11]["Lower Bound"] = -1.0
e["Variables"][11]["Upper Bound"] = +1.0

### Configuring NAF hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Target Learning Rate"] = 0.0001
e["Solver"]["Experiences Between Policy Updates"] = 5
e["Solver"]["Mini Batch Strategy"] = "Prioritized"

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

## Defining Critic Info

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch Size"] = 32

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

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 900

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
