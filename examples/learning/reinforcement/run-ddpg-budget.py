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
    reward = reward - abs(budget)*5.0
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

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent/DDPG"
e["Solver"]["Agent"]["Episodes Per Generation"] = 1
e["Solver"]["Agent"]["Experience Limit"] = maxSteps

### Defining the configuration of replay memory

e["Solver"]["Replay Memory"]["Start Size"] =   10000
e["Solver"]["Replay Memory"]["Maximum Size"] = 100000

## Defining Critic Configuration

e["Solver"]["Critic"]["Optimization Steps"] = 50
e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.01
e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Mini Batch Size"] = 64

## Defining Policy Configuration

e["Solver"]["Policy"]["Optimization Steps"] = 5
e["Solver"]["Policy"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Policy"]["Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0
e["Solver"]["Policy"]["Optimizer"]["Eta"] = 0.0001
e["Solver"]["Policy"]["Mini Batch Size"] = 16

### Defining the shape of the critic neural network

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Node Count"] = 3
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Node Count"] = 1
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001

e["Solver"]["Critic"]["Normalization Steps"] = 32

### Defining the shape of the policy neural network

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Linear"

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "ReLU"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Alpha"] = 0.0

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "ReLU"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Alpha"] = 0.0

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Node Count"] = 2
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Logistic" 
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001

e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = [ 10.0, 10.0 ]

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Target Average Reward"] = 0

### Setting file output configuration

e["File Output"]["Frequency"] = 10000
#e["Console Output"]["Verbosity"] = "Silent"

### Running Experiment

#k["Conduit"]["Type"] = "Concurrent"
#k["Conduit"]["Concurrent Jobs"] = 5
k.run(e)