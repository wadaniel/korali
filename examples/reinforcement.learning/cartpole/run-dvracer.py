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
    default=50,
    type=int,
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
    type=float,
    required=False)
args = parser.parse_args()

print("Running Cartpole example with arguments:")
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = [ [ -10.0 ], [  10.0 ] ]
e["Problem"]["Environment Function"] = env

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


### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Discrete / dVRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Mini Batch"]["Size"] = 32

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 256
e["Solver"]["Experience Replay"]["Maximum Size"] = 65536

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

### Configuring the neural network and its hidden layers

e["Solver"]['Time Sequence Length'] = 4                                         

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Recurrent/LSTM"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Depth"] = 1                  
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 64

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGenerations

### Setting file output configuration

e["File Output"]["Enabled"] = False
e["Console Output"]["Verbosity"] = "Detailed"

### Running Experiment

k.run(e)
