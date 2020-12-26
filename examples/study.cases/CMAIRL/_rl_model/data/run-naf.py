#!/usr/bin/env python3
import os
import sys
sys.path.append('../_model')
from env import *

target = 0.0
outfile = "observations-naf.dat"

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

envp = lambda s : env(s,target)

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = envp
e["Problem"]["Training Reward Threshold"] = 490
e["Problem"]["Policy Testing Episodes"] = 10
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

### Configuring NAF hyperparameters

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Target Learning Rate"] = 0.01
e["Solver"]["Experiences Between Policy Updates"] = 5
e["Solver"]["Covariance Scaling"] = 0.001
e["Solver"]["Mini Batch Strategy"] = "Prioritized"

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] =   1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

## Defining Q-Network

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 1e-4
e["Solver"]["Critic"]["Mini Batch Size"] = 256

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 490
e["Solver"]["Termination Criteria"]["Max Generations"] = 2500

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

### Recording Observations

print('[Korali] Done training. Now running learned policy to produce observations.')

states = []
actions = []

cart = CartPole(0.0)

state = cart.getState().tolist()
done = False
step = 0
while not done and step < 100:
 
 action = e.getAction(state)
 
 states.append(state)
 actions.append(action)
 
 cart.advance(action)

 reward = cart.getReward()
 step = step + 1

 state = cart.getState().tolist()

### Running Experiment

k.run(e)

print('[Korali] Finished recording observations. Writing file {}..'.format(outfile))
with open(outfile, 'w') as f:
    for i in range(len(states)):
        f.write(str(actions[i][0]))
        f.write(", ")
        f.write(", ".join(str(s) for s in states[i]))
        f.write("\n")
