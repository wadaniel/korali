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

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "Agent / Discrete / DQN"
e["Solver"]["Experiences Between Policy Updates"] = 1

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

### Defining probability of taking a random action (epsilon)

e["Solver"]["Random Action Probability"]["Initial Value"] = 1.0
e["Solver"]["Random Action Probability"]["Target Value"] = 0.01
e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.10

## Defining Q-Critic and Action-selection (policy) optimizers

e["Solver"]["Critic"]["Mini Batch Size"] = 32
e["Solver"]["Critic"]["Learning Rate"] = 0.001
e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Target Update Delay"] = 500

### Defining the shape of the neural network

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 450

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)

### Now generating movie with the learned policy

movieFile = './movie'
cart = gym.wrappers.Monitor(cart, movieFile, force=True)
maxSteps = 1000

print('[Korali] Done training. Now running learned policy to produce the movie.')

state = cart.reset().tolist()
done = False
step = 0
while not done and step < 100:
 action = int(e.getAction(state)[0])
 print('[Korali] Step ' + str(step) + ' - State: ' + str(state) + ' - Action: ' + str(action), end = '')
 state, reward, done, info = cart.step(action)
 print('- Reward: ' + str(reward))
 step = step + 1

print('[Korali] Finished. Movie stored in the folder: ' + movieFile)