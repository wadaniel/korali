#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--maxGenerations',
    help='Maximum Number of generations to run',
    default=1000,
    required=False)    
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=1e-3,
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
e["Problem"]["Training Reward Threshold"] = 1e9
e["Problem"]["Policy Testing Episodes"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Position Y"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Cart Velocity X"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Cart Velocity Y"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Cart Acceleration X"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Cart Acceleration Y"
e["Variables"][5]["Type"] = "State"

e["Variables"][6]["Name"] = "Force"
e["Variables"][6]["Type"] = "Action"
e["Variables"][6]["Lower Bound"] = -3.0
e["Variables"][6]["Upper Bound"] = +3.0
e["Variables"][6]["Initial Exploration Noise"] = 1.0

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 1

e["Solver"]["Experience Replay"]["Start Size"] = 10000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = float(args.learningRate)
e["Solver"]["Mini Batch"]["Size"] = 128

e["Solver"]["State Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000

### Configuring the neural network and its hidden layers

e["Solver"]["Policy"]["Distribution"] = "Normal"
#e["Solver"]["Policy"]["Distribution"] = "Squashed Normal"
#e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
#e["Solver"]["Policy"]["Distribution"] = "Truncated Normal"
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = 250
e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 1e9

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

### Now we run a few test samples and check their reward

#e["Solver"]["Mode"] = "Testing"
#e["Solver"]["Testing"]["Sample Ids"] = list(range(5))

#k.run(e)

#averageTestReward = np.average(e["Solver"]["Testing"]["Reward"])
#print("Average Reward: " + str(averageTestReward))
#if (averageTestReward < 150):
# print("Cartpole example did not reach minimum testing average.")
# exit(-1)

