#!/usr/bin/env python3
import os
import sys
sys.path.append('./_environment')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

populationSize = 8

lEnv = lambda s : env(s,populationSize)

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lEnv
e["Problem"]["Training Reward Threshold"] = 400
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 5


for i in range(populationSize):
    e["Variables"][i*3]["Name"] = "Position X"
    e["Variables"][i*3]["Type"] = "State"

    e["Variables"][i*3+1]["Name"] = "Position Y"
    e["Variables"][i*3+1]["Type"] = "State"

    e["Variables"][i*3+2]["Name"] = "Evaluation"
    e["Variables"][i*3+2]["Type"] = "State"

i = 3*populationSize
e["Variables"][i]["Name"] = "Mean X"
e["Variables"][i]["Type"] = "Action"
e["Variables"][i]["Lower Bound"] = -10.0
e["Variables"][i]["Upper Bound"] = +10.0
e["Variables"][i]["Initial Exploration Noise"] = 1.0

i += 1
e["Variables"][i]["Name"] = "Mean Y"
e["Variables"][i]["Type"] = "Action"
e["Variables"][i]["Lower Bound"] = -10.0
e["Variables"][i]["Upper Bound"] = +10.0
e["Variables"][i]["Initial Exploration Noise"] = 1.0


### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 10
e["Solver"]["Episodes Per Generation"] = 1

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-3
e["Solver"]["Mini Batch"]["Size"] = 128

e["Solver"]["State Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000
e["Solver"]["Policy"]["Distribution"] = "Normal"

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "AdaBelief"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = 1000

### If this is test mode, run only a couple generations
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  e["Solver"]["Termination Criteria"]["Max Generations"] = 5
  
### Setting file output configuration

e["File Output"]["Enabled"] = True

### Running Experiment

k.run(e)
