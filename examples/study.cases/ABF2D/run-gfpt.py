#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

parser = argparse.ArgumentParser(prog='GFPT', description='Runs the GFPT algorithm on ABF2D')
parser.add_argument('--dir', help='directory of result files',  default='_result', required=False)
args = parser.parse_args()
setResultsDir(args.dir)
    
### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 40.0
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 1

### Defining state variables

e["Variables"][0]["Name"] = "Swimmer 1 - Pos X"
e["Variables"][1]["Name"] = "Swimmer 1 - Pos Y"
e["Variables"][2]["Name"] = "Swimmer 2 - Pos X"
e["Variables"][3]["Name"] = "Swimmer 2 - Pos Y"

### Defining action variables

e["Variables"][4]["Name"] = "Magnet Rotation X"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -1.0
e["Variables"][4]["Upper Bound"] = +1.0
e["Variables"][4]["Exploration Sigma"] = 0.5

e["Variables"][5]["Name"] = "Magnet Rotation Y"
e["Variables"][5]["Type"] = "Action"
e["Variables"][5]["Lower Bound"] = -1.0
e["Variables"][5]["Upper Bound"] = +1.0
e["Variables"][5]["Exploration Sigma"] = 0.5

e["Variables"][6]["Name"] = "Magnet Intensity"
e["Variables"][6]["Type"] = "Action"
e["Variables"][6]["Lower Bound"] = +0.0
e["Variables"][6]["Upper Bound"] = +2.0
e["Variables"][6]["Exploration Sigma"] = 0.5

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Time Sequence Length"] = 4
e["Solver"]["Experiences Per Generation"] = 600
e["Solver"]["Experiences Between Policy Updates"] = 200
e["Solver"]["Cache Persistence"] = 20

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] =   1024
e["Solver"]["Experience Replay"]["Maximum Size"] = 32768
e["Solver"]["Experience Replay"]["Serialization Frequency"] = 100

### Configuring Mini Batch

e["Solver"]["Mini Batch Size"] = 128
e["Solver"]["Mini Batch Strategy"] = "Uniform"

## Defining Critic and Policy Configuration

e["Solver"]["Critic"]["Learning Rate"] = 0.001  
e["Solver"]["Policy"]["Learning Rate"] = 0.0001
e["Solver"]["Policy"]["Target Accuracy"] = 0.0001
e["Solver"]["Policy"]["Optimization Candidates"] = 24

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 40.0

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 100
e["File Output"]["Path"] = "_results"

### Running Experiment

k.run(e)
