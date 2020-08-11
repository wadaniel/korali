#!/usr/bin/env python3
import os
import sys
import math
import gym

######## Defining Environment Storage

maxSteps = 5
initialBudget = 25

####### Defining Problem's environment

def env(s):

 # Initializing environment
 budget = initialBudget
 step = 0

 while step < maxSteps:

  # Storing current budget as state
  s["State"] = [ budget ]
  
  # Getting new action
  s.update()
  
  # Reading action
  action0 = s["Action"][0]
  action1 = s["Action"][1]
  print('Python Action: ' + str(action0) + ' ' + str(action1) ) 
  
  # Reward = action, if not bigger than state
  val0 = action0 - 5.0
  val1 = action1 - 2.5
  reward = -val0*val0 -val1*val1
  
  # Calculating remaining budget
  spenditure = val0 + val1
  budget = budget - spenditure
  
  # If budget is negative (overdraft), set to zero and punish
  if (budget < 0): 
    reward = reward - abs(budget)
    budget = 0
    
  # Storing Reward
  s["Reward"] = reward
   
  # Advancing step counter
  step = step + 1
  
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = env

e["Variables"][0]["Name"] = "Budget"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Spenditure"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = -6.0
e["Variables"][1]["Upper Bound"] = +6.0

e["Variables"][2]["Name"] = "Spenditure"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Lower Bound"] = -6.0
e["Variables"][2]["Upper Bound"] = +6.0

### Defining noise to add to the action

e["Variables"][1]["Exploration Noise"]["Enabled"] = True
e["Variables"][1]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal"
e["Variables"][1]["Exploration Noise"]["Distribution"]["Mean"] = 0.0
e["Variables"][1]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.1
e["Variables"][1]["Exploration Noise"]["Theta"] = 0.05

e["Variables"][2]["Exploration Noise"]["Enabled"] = True
e["Variables"][2]["Exploration Noise"]["Distribution"]["Type"] = "Univariate/Normal"
e["Variables"][2]["Exploration Noise"]["Distribution"]["Mean"] = 0.0
e["Variables"][2]["Exploration Noise"]["Distribution"]["Standard Deviation"] = 0.1
e["Variables"][2]["Exploration Noise"]["Theta"] = 0.05

e["Solver"]["Type"] = "Agent/DDPG"

### Defining Mini-batch and DDPG configuration 

e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Optimization Steps Per Generation"] = 1
e["Solver"]["Agent History Size"] = maxSteps
e["Solver"]["Mini Batch Size"] = 32
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Adoption Rate"] = 0.001

### Defining the configuration of replay memory

e["Solver"]["Replay Memory"]["Start Size"] = 32
e["Solver"]["Replay Memory"]["Maximum Size"] = 100000
e["Solver"]["Replay Memory"]["Replacement Policy"] = "Least Recently Added"

## Defining Actor and Critic optimizers

e["Solver"]["Actor Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Actor Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0
e["Solver"]["Actor Optimizer"]["Eta"] = 0.000001

e["Solver"]["Critic Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic Optimizer"]["Eta"] = 0.005

### Defining the shape of the critic neural network

e["Solver"]["Critic Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic Neural Network"]["Layers"][0]["Node Count"] = 3
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

e["Solver"]["Batch Normalization"]["Correction Steps"] = 32

### Defining the shape of the actor neural network

e["Solver"]["Actor Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Actor Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Actor Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Actor Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Actor Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "ReLU"
e["Solver"]["Actor Neural Network"]["Layers"][1]["Activation Function"]["Alpha"] = 0.0
e["Solver"]["Actor Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Actor Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Actor Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "ReLU"
e["Solver"]["Actor Neural Network"]["Layers"][2]["Activation Function"]["Alpha"] = 0.0
e["Solver"]["Actor Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Actor Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Actor Neural Network"]["Layers"][3]["Node Count"] = 2
e["Solver"]["Actor Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Logistic" 
e["Solver"]["Actor Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Actor Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.0000001

e["Solver"]["Actor Neural Network"]["Output Scaling"] = [ 10.0, 10.0 ]

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Reward"] = 0

### Setting file output configuration

e["File Output"]["Frequency"] = 10000
#e["Console Output"]["Verbosity"] = "Silent"

### Running Experiment

#k["Conduit"]["Type"] = "Concurrent"
#k["Conduit"]["Concurrent Jobs"] = 5
k.run(e)