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
  reward = -2*(val0*val0 + val1*val1)
  
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

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Budget"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Spenditure"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = 10.0
e["Variables"][1]["Noise Sigma"] = 0.10

e["Variables"][2]["Name"] = "Spenditure"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Lower Bound"] = 0.0
e["Variables"][2]["Upper Bound"] = 10.0
e["Variables"][2]["Noise Sigma"] = 0.10

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / ACER / Continuous"
e["Solver"]["Importance Weight Truncation"] = 5.0
e["Solver"]["Trust Region Divergence Constraint"] = 1.0
e["Solver"]["Trajectory Size"] = 5
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Off Policy Updates"] = 8
e["Solver"]["Optimization Steps Per Trajectory"] = 1

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] =   500
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

## Defining Critic Configuration

e["Solver"]["Critic"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Critic"]["Optimizer"]["Eta"] = 0.001

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001 

## Defining Policy Configuration

e["Solver"]["Policy"]["Optimizer"]["Type"] = "Optimizer/Adam"
e["Solver"]["Policy"]["Optimizer"]["Termination Criteria"]["Min Gradient Norm"] = -1.0
e["Solver"]["Policy"]["Optimizer"]["Eta"] = 0.00005
e["Solver"]["Policy"]["Adoption Rate"] = 0.995
e["Solver"]["Policy"]["Sample Population"] = 5

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Alpha"] = 0.0

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Alpha"] = 0.0

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Node Count"] = 4
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Logistic" 
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.000000001

e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = [ 10.0, 10.0 ]

### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = -0.5
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = -0.5

### Setting file output configuration

e["File Output"]["Frequency"] = 10000
#e["Console Output"]["Verbosity"] = "Silent"

### Running Experiment

k.run(e)
