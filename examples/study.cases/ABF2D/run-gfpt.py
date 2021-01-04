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
e["Problem"]["Training Reward Threshold"] = 42.0
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 200

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
e["Variables"][4]["Exploration Sigma"]["Initial"] = 0.5
e["Variables"][4]["Exploration Sigma"]["Final"] = 0.05
e["Variables"][4]["Exploration Sigma"]["Annealing Rate"] = 1e-5

e["Variables"][5]["Name"] = "Magnet Rotation Y"
e["Variables"][5]["Type"] = "Action"
e["Variables"][5]["Lower Bound"] = -1.0
e["Variables"][5]["Upper Bound"] = +1.0
e["Variables"][5]["Exploration Sigma"]["Initial"] = 0.5
e["Variables"][5]["Exploration Sigma"]["Final"] = 0.05
e["Variables"][5]["Exploration Sigma"]["Annealing Rate"] = 1e-5

e["Variables"][6]["Name"] = "Magnet Intensity"
e["Variables"][6]["Type"] = "Action"
e["Variables"][6]["Lower Bound"] = +0.0
e["Variables"][6]["Upper Bound"] = +2.0
e["Variables"][6]["Exploration Sigma"]["Initial"] = 0.5
e["Variables"][6]["Exploration Sigma"]["Final"] = 0.05
e["Variables"][6]["Exploration Sigma"]["Annealing Rate"] = 1e-5

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 10
e["Solver"]["Cache Persistence"] = 200
e["Solver"]["Learning Rate"] = 0.001

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 65536
e["Solver"]["Experience Replay"]["Serialization"]["Frequency"] = 100

### Configuring the Remember-and-Forget Experience Replay algorithm

e["Solver"]["Experience Replay"]["REFER"]["Enabled"] = True
e["Solver"]["Experience Replay"]["REFER"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["REFER"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["REFER"]["Initial Beta"] = 0.6
e["Solver"]["Experience Replay"]["REFER"]["Annealing Rate"] = 5e-7

### Configuring Mini Batch

e["Solver"]["Mini Batch Size"] = 128
e["Solver"]["Mini Batch Strategy"] = "Uniform"

## Defining Critic and Policy Configuration

e["Solver"]["Critic"]["Advantage Function Population"] = 12
e["Solver"]["Policy"]["Learning Rate Scale"] = 0.1
e["Solver"]["Policy"]["Target Accuracy"] = 0.01
e["Solver"]["Policy"]["Optimization Candidates"] = 24

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 64

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 42.0

### Setting console/file output configuration

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 100
e["File Output"]["Path"] = "_results"

### Running Experiment

k.run(e)
