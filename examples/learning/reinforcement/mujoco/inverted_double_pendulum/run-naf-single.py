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
e["Problem"]["Action Repeat"] = 1
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
e["Variables"][11]["Lower Bound"] = -10.0
e["Variables"][11]["Upper Bound"] = +10.0

### Configuring NAF hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Target Learning Rate"] = 0.001
e["Solver"]["Optimization Steps Per Update"] = 100
e["Solver"]["Experiences Between Agent Trainings"] = 5
e["Solver"]["Mini Batch Strategy"] = "Prioritized"

e["Solver"]["Random Action Probability"]["Initial Value"] = 0.3
e["Solver"]["Random Action Probability"]["Target Value"] = 0.001
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.03

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

## Defining Q-Network

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 1e-4
e["Solver"]["Critic"]["Mini Batch Size"] = 32

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 750
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 900

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
