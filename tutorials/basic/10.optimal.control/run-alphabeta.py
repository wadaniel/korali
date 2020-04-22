#!/usr/bin/env python3
import math
import numpy as np

######## Defining Problem's Constants

N = 3 # Number of stages
initialX = 1.0 # Initial value of X
alpha = 0.75 # Alpha
beta = 0.50 # Beta
intervals = 200 # How fine will we discretize the variable space

######## Defining Problem's Formulae

# Reward Function g(y)
def g(y):
 return np.cos(1 + 23 * y)

# Reward Function h(x-y)
def h(v):
 return np.sin(10 * v)
 
# Reward function to optimize
def rewardFunction(k):

  # Initialize X as per problem's specifications
  currentX = initialX 

  # Recovering the value of x, based on policy's choices  
  for p in k["Policy"]:
    y = p[0]
    
    # Check if the value of y is valid 
    if (y > currentX):
     # If Yi greater than Xi, then this is not a feasible policy, returning infinite cost 
     k["Cost Evaluation"] = math.inf
     return 
     
    # After this loop, this variable will contain the value of X for the current recursion stage, which we need.
    x = currentX 
    
    # Updating the value of x, based on the policy choice
    currentX = alpha*y + beta*(x-y)
     
  # Since Korali minimizes cost, we maximize the reward by negating it
  reward = g(y) + h(x-y)
  k["Cost Evaluation"] = -reward

######## Configuring Korali Experiment

import korali
  
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Optimal Control"
e["Problem"]["Cost Function"] = rewardFunction

# Defining problem variables to discretize.
e["Variables"][0]["Name"] = "Y"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = initialX
e["Variables"][0]["Interval Count"] = intervals

# Configuring the discretizer solver's parameters
e["Solver"]["Type"] = "Dijkstra"
e["Solver"]["Termination Criteria"]["Recursion Depth"] = N

######## Running Korali and printing results

k = korali.Engine()
k.run(e)

print('Best Policy:     ' + str(e["Results"]["Optimal Policy"]))
print('Optimal Reward:  ' + str(-e["Results"]["Policy Evaluation"]))
