#!/usr/bin/env python3
import os
import sys
sys.path.append('../_optimization_model/_rl_model')
from env import *
from evalenv import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

target = 0.5
envp = lambda s : env(s,target)

### Defining the Cartpole problem's configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = envp
e["Problem"]["Training Reward Threshold"] = 490
e["Problem"]["Testing Frequency"] = 100
e["Problem"]["Policy Testing Episodes"] = 25
e["Problem"]["Actions Between Policy Updates"] = 5

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

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 5
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Cache Persistence"] = 2000

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] =   2048
e["Solver"]["Experience Replay"]["Maximum Size"] = 32768

e["Solver"]["Experience Replay"]["REFER"]["Enabled"] = True
e["Solver"]["Experience Replay"]["REFER"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["REFER"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["REFER"]["Initial Beta"] = 0.3
e["Solver"]["Experience Replay"]["REFER"]["Annealing Rate"] = 5e-7

## Defining Neural Network Configuration for Policy and Critic into Critic Container

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

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 499
e["Solver"]["Termination Criteria"]["Max Generations"] = 5000
e["Solver"]["Termination Criteria"]["Testing"]["Average Reward Increment"] = 1.0

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

e["Problem"]["Custom Settings"]["Record Observations"] = "False"

k.run(e)


### Evaluate Policy
    
e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = [0]
e["Problem"]["Environment Function"] = evalenv

k.run(e)

suml2error = e["Solver"]["Testing"]["Reward"][0]

print("[Korali] Finished testing (p {0} error {1}.".format(target, suml2error))
