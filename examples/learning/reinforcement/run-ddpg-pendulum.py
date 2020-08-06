#!/usr/bin/env python3
import os
import sys
import math
import gym

######## Defining Environment Storage

pendulum = gym.make('Pendulum-v0').unwrapped
maxSteps = 200

####### Defining Problem's environment

def env(s):

 # Initializing environment
 seed = s["Sample Id"]
 pendulum.seed(seed)
 s["State"] = pendulum.reset().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Reading action
  action = s["Action"]
  action[0] = action[0] * 2
  #print(action)
      
  # Performing the action
  state, reward, done, info = pendulum.step(action)

  # Storing Reward
  s["Reward"] = reward
   
  # Storing New State
  s["State"] = state.tolist()
  
  # Advancing step counter
  step = step + 1
  
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Omega"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Torque"
e["Variables"][3]["Type"] = "Action"

### Defining noise to add to the action

e["Variables"][3]["Exploration Noise"]["Random Variable"]["Type"] = "Univariate/Normal"
e["Variables"][3]["Exploration Noise"]["Random Variable"]["Mean"] = 0.0
e["Variables"][3]["Exploration Noise"]["Random Variable"]["Standard Deviation"] = 0.10
e["Variables"][3]["Exploration Noise"]["Theta"] = 0.15

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "Agent/DDPG"

### Defining Mini-batch and DDPG configuration 

e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Optimization Steps Per Generation"] = 10
e["Solver"]["Agent History Size"] = maxSteps
e["Solver"]["Mini Batch Size"] = 64
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Adoption Rate"] = 0.001 

### Defining the configuration of replay memory

e["Solver"]["Replay Memory"]["Start Size"] = 2000
e["Solver"]["Replay Memory"]["Maximum Size"] = 500000
e["Solver"]["Replay Memory"]["Replacement Policy"] = "Least Recently Added"

## Defining Actor and Critic optimizers

e["Solver"]["Actor Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Actor Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0
e["Solver"]["Actor Optimizer"]["Eta"] = 0.0001

e["Solver"]["Critic Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic Optimizer"]["Eta"] = 0.01

### Defining the shape of the critic neural network

e["Solver"]["Critic Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic Neural Network"]["Layers"][0]["Node Count"] = 4
e["Solver"]["Critic Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Critic Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Critic Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Critic Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Critic Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Critic Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False 
e["Solver"]["Critic Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000001

e["Solver"]["Batch Normalization"]["Correction Steps"] = 32

### Defining the shape of the actor neural network

e["Solver"]["Actor Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][0]["Node Count"] = 3
e["Solver"]["Actor Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Actor Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Actor Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Actor Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Actor Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Actor Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Actor Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Actor Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Actor Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Actor Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Tanh" 
e["Solver"]["Actor Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Actor Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000001

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Reward"] = -100

### Setting file output configuration

e["File Output"]["Frequency"] = 1000

### Running Experiment

#k["Conduit"]["Type"] = "Concurrent"
#k["Conduit"]["Concurrent Jobs"] = 5
k.run(e)

###### Now running the pendulum experiment with Korali's help

state = pendulum.reset().tolist()
step = 0
done = False

while not done and step < maxSteps:
 action = e.getAction(state)
 print('Step ' + str(step) + ' - State: ' + str(state) + ' - Action: ' + str(action), end = '')
 state, reward, done, info = pendulum.step(action)
 print('- Reward: ' + str(reward))
 step = step + 1