#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--maxExperiences',
    help='Maximum Number of experiences to run',
    default=1000000,
    type=int,
    required=False)    

args = parser.parse_args()

print("Running Cartpole example with arguments:")
print(args)
####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Testing Frequency"] = 100
e["Problem"]["Policy Testing Episodes"] = 25

e["Variables"][0]["Name"] = "Inventory"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Time"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Sell Fraction"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Lower Bound"] = .0
e["Variables"][2]["Upper Bound"] = 1.0
e["Variables"][2]["Initial Exploration Noise"] = 0.1

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 10

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 524288
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"]= 0.3

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-3
e["Solver"]["Mini Batch"]["Size"] = 128
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/Tanh"


### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.maxExperiences

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Frequency"] = 100

### Running Experiment

k.run(e)
