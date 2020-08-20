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
  #print(action[0])
      
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
e["Variables"][3]["Lower Bound"] = -2.0
e["Variables"][3]["Upper Bound"] = +2.0

### Defining noise to add to the action

e["Variables"][3]["Exploration Noise"]["Enabled"] = True
e["Variables"][3]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal"
e["Variables"][3]["Exploration Noise"]["Distribution"]["Mean"] = 0.0
e["Variables"][3]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.1
e["Variables"][3]["Exploration Noise"]["Theta"] = 0.05

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent/DDPG"
e["Solver"]["Agent"]["Episodes Per Generation"] = 5
e["Solver"]["Agent"]["Experience Limit"] = maxSteps

### Defining the configuration of replay memory

e["Solver"]["Replay Memory"]["Start Size"] =   10000
e["Solver"]["Replay Memory"]["Maximum Size"] = 100000

## Defining Critic Configuration

e["Solver"]["Critic"]["Optimization Steps"] = 50
e["Solver"]["Critic"]["Update Algorithm"] = "Q-Learning"
e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.001
e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Mini Batch Size"] = 32

## Defining Policy Configuration

e["Solver"]["Policy"]["Optimization Steps"] = 200
e["Solver"]["Policy"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Policy"]["Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0
e["Solver"]["Policy"]["Optimizer"]["Eta"] = 0.00001
e["Solver"]["Policy"]["Mini Batch Size"] = 1000

### Defining the shape of the critic neural network

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 4
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 16
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Weight Initialization Scaling"] = 0.000001

e["Solver"]["Normalization Steps"] = 32

### Defining the shape of the actor neural network

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 3
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "ReLU"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Alpha"] = 0.0
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh" 
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Weight Initialization Scaling"] = 0.000001

e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = [ 2.0 ]

### Defining Termination Criteria

e["Solver"]["Average Training Reward Threshold"] = -2.0
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = -0.1

### Setting file output configuration

e["File Output"]["Frequency"] = 5000

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