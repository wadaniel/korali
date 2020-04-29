#!/usr/bin/env python3
import math
import numpy as np

######## Defining Problem's Configuration

initialBudget = 1.0 # Initial Budget (x)
granularity = 0.005 # Discretization factor: how fine will we discretize the variable space

# Defining list of cost functions, one per stage 
evalFunctions = [
  lambda x : 2*x,
  lambda x : -30*x*x + 17.63*x
  ]

# Environment that provides the state/action loop
def environment(k):

  # Setting initial budget
  budget = initialBudget
  
  # Initializing reward
  k["Reward"] = 0
  
  # For each of the cost functions (stages)
  for Fc in evalFunctions:

   # Updating state with the value of the current (remaining) budget
   k["State"] = [ budget ]  
  
   # Get back to Korali to obtain the next action to perform
   k.update()

   # Getting Action (how much to spend in this step)
   spenditure = k["Action"][0] 
  
   # If spending decision is higher than remaining budget, then this is an inviable decision.
   if (spenditure > budget):
    k["Reward"] = -math.inf
    return 
    
   # Else, we substract the spenditure from the budget
   budget = budget - spenditure
   
   # Then add the function calculation to the reward.
   k["Reward"] = k["Reward"] + Fc(spenditure)

######## Configuring Korali Experiment

import korali

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = environment

# Defining problem's state.
e["Variables"][0]["Name"] = "Budget"
e["Variables"][0]["Type"] = "State"
e["Variables"][0]["Parameter Space"] = "Discrete"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = initialBudget
e["Variables"][0]["Granularity"] = granularity

# Defining problem's actions.
e["Variables"][1]["Name"] = "Spenditure"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Parameter Space"] = "Discrete"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = initialBudget
e["Variables"][1]["Granularity"] = granularity

# Configuring the solver
e["Solver"]["Type"] = "Learner/QTable"
e["Solver"]["Q Update Algorithm"] = "Q-Learning"
e["Solver"]["Initial Q Value"] = 10

# Configuring Output
e["Console Output"]["Frequency"] = 5000
e["File Output"]["Enabled"] = False

######## Running Korali and printing results

k = korali.Engine()
k.run(e)

print('Best Policy:     ' + str(e["Results"]["Optimal Policy Actions"]))
print('Optimal Reward:  ' + str(e["Results"]["Optimal Reward"]))
