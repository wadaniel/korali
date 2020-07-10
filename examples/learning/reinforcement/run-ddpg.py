#!/usr/bin/env python3
import os
import sys
import math
import gym

######## Defining Environment Storage

car = gym.make('MountainCarContinuous-v0').unwrapped
maxSteps = 1000

####### Defining Problem's environment

def env(s):

 # Initializing environment
 print(car.reset().tolist())
 s["State"] = car.reset().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Reading action
  action = s["Action"] 
    
  # Performing the action
  state, reward, done, info = car.step(action)

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

e["Variables"][0]["Name"] = "Car Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Car Speed"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Force"
e["Variables"][2]["Type"] = "Action"

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "Agent/DDPG"

### Defining Mini-batch and DDPG configuration 

e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Optimization Steps Per Generation"] = 10

e["Solver"]["Agent History Size"] = 1000
e["Solver"]["Mini Batch Size"] = 32
e["Solver"]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Batch Normalization"]["Correction Steps"] = 64

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Adoption Rate"] = 0.0001
e["Solver"]["Exploration Noise"]["Standard Deviation"] = 0.001

### Defining the configuration of replay memory
 
e["Solver"]["Replay Memory"]["Start Size"] = 5000
e["Solver"]["Replay Memory"]["Maximum Size"] = 50000
e["Solver"]["Replay Memory"]["Replacement Policy"] = "Least Recently Added"

## Defining Actor and Critic optimizers

e["Solver"]["Actor Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Actor Optimizer"]["Eta"] = 0.05

e["Solver"]["Critic Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic Optimizer"]["Eta"] = 0.05

### Defining the shape of the critic neural network

e["Solver"]["Critic Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Critic Neural Network"]["Layers"][0]["Node Count"] = 3
e["Solver"]["Critic Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Identity"

e["Solver"]["Critic Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Critic Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Critic Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Critic Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Critic Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Critic Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Critic Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Identity" 

### Defining the shape of the actor neural network

e["Solver"]["Actor Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Actor Neural Network"]["Layers"][0]["Node Count"] = 2
e["Solver"]["Actor Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Identity"

e["Solver"]["Actor Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Actor Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Actor Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Actor Neural Network"]["Layers"][2]["Type"] = "Dense"
e["Solver"]["Actor Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Actor Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Actor Neural Network"]["Layers"][3]["Type"] = "Output"
e["Solver"]["Actor Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Actor Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Logistic" 

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Reward"] = 900

### Setting file output configuration

e["File Output"]["Frequency"] = 1
                                      
###### Loading any previous results

found = e.loadState('_korali_result/latest')

if (found == False):
 k.run(e)
else:
 print('Found pre-trained experiment') 

###### Now running the cartpole experiment with Korali's help

state = cart.reset().tolist()
step = 0
done = False

while not done and step < maxSteps:
 action = int(e.getAction(state)[0])
 print('Step ' + str(step) + ' - State: ' + str(state) + ' - Action: ' + str(action))
 state, reward, done, info = cart.step(action)
 step = step + 1
