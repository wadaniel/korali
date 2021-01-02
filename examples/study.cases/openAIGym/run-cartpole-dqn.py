#!/usr/bin/env python3
import os
import sys 
import gym
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = [ [ 0.0 ], [ 1.0 ] ]
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 800
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 1

### Adding custom setting to run the environment without saving a movie during training
e["Problem"]["Custom Settings"]["Save Movie"] = "Disabled"

### Defining variables

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Angular Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Push Direction"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Exploration Sigma"]["Initial"] = 1.0
e["Variables"][4]["Exploration Sigma"]["Final"] = 1.0
e["Variables"][4]["Exploration Sigma"]["Annealing Rate"] = 0.0

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / GFPT"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Time Sequence Length"] = 1
e["Solver"]["Experiences Per Generation"] = 500
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Cache Persistence"] = 10
e["Solver"]["Discount Factor"] = 0.99

### Defining the configuration of replay memory

e["Solver"]["Mini Batch Size"] = 32
e["Solver"]["Mini Batch Strategy"] = "Uniform"
e["Solver"]["Experience Replay"]["Start Size"] =   1024
e["Solver"]["Experience Replay"]["Maximum Size"] = 32768
e["Solver"]["Experience Replay"]["Serialization Frequency"] = 10

## Defining Critic and Policy Configuration

e["Solver"]["Learning Rate"] = 0.01
e["Solver"]["Policy"]["Learning Rate Scale"] = 1.0
e["Solver"]["Critic"]["Advantage Function Population"] = 12
e["Solver"]["Policy"]["Target Accuracy"] = 0.001
e["Solver"]["Policy"]["Optimization Candidates"] = 12

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

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 800

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

### Now generating movie with the learned policy

print('[Korali] Done with training. Now running learned policy to produce the movie.')

moviePath = './_movie'
e["Problem"]["Custom Settings"]["Save Movie"] = "Enabled"
e["Problem"]["Custom Settings"]["Movie Path"] = moviePath
e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = [ 0 ]
k.run(e)

print('[Korali] Finished. Movie stored in the folder : ' + movieFile)