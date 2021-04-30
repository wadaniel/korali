#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
import math
import json
from env import *


"""Note:
    - COMPILE KORALI with COSREWARD 
"""

####### Load observations

obsfile = "observations-t-0.0.json"
obsstates = []
obsactions = []
with open(obsfile, 'r') as infile:
    obsjson = json.load(infile)
    obsstates = obsjson["States"]
    obsactions = obsjson["Actions"]

### Compute Feauters from states

maxFeatures = [-math.inf]
obsfeatures = []
for trajectory in obsstates:
    features = []
    for state in trajectory:
        # Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
        feature1 = state[2]
        features.append([feature1]) 
        
        if(maxFeatures[0] < feature1):
            maxFeatures[0] = feature1
    obsfeatures.append(features)
    
print("Total observed trajectories: {}/{}".format(len(obsstates), len(obsactions)))
print("Max feature values found in observations:")
print(maxFeatures)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = cosenv
e["Problem"]["Training Reward Threshold"] = 501
e["Problem"]["Policy Testing Episodes"] = 1
#e["Problem"]["Actions Between Policy Updates"] = 1
e["Problem"]["Observations"]["States"] = obsstates
e["Problem"]["Observations"]["Actions"] = obsactions
e["Problem"]["Observations"]["Features"] = obsfeatures

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Angular Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Force"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -10.0
e["Variables"][4]["Upper Bound"] = +10.0
e["Variables"][4]["Initial Exploration Noise"] = 0.1

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Experiences Between Reward Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 1

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 65536

## Defining Neural Network Configuration for Policy and Critic into Critic Container

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Rewardfunction Learning Rate"] = 5e-5
e["Solver"]["Mini Batch Size"] = 256
e["Solver"]["Updates Between Reward Rescaling"] = 0

### IRL related configuration

e["Solver"]["Updates Between Reward Rescaling"] = 0 # No reward rescaling
e["Solver"]["Demonstration Batch Size"] = 100
e["Solver"]["Background Batch Size"] = 20
e["Solver"]["Use Fusion Distribution"] = True
e["Solver"]["Experiences Between Partition Function Statistics"] = 100000

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

#e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 450
e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 100
#e["File Output"]["Path"] = '_korali_results_cos_z20_combo_wlin'
e["File Output"]["Path"] = '_korali_results_cos_test'

### Running Experiment

k.run(e)
