#!/usr/bin/env python3
import math
import numpy as np

######## Defining Problem's Constants

N = 3 # Number of stages
initialX = 1.0 # Initial value of X
alpha = 0.75 # Alpha
beta = 0.50 # Beta
granularity = 0.005 # Discretization factor: how fine will we discretize the variable space

######## Defining Problem's Formulae

# Reward Function g(y)
def g(y):
 return np.cos(1 + 23 * y)

# Reward Function h(x-y)
def h(v):
 return np.sin(10 * v)
 
# Environment that provides the state/action loop
def environment(k):

  # Initialize X as per problem's specifications
  x = initialX 

  # Initializing reward
  k["Reward"] = 0.0    
  
  # Run the policy for N stages
  for i in range(N):

   # Setting current State
   k["State"] = [ x ]
   
   # Get back to Korali to obtain the next action to perform
   k.update()
  
   # Getting Action (the value of Y)
   y = k["Action"][0] 
  
   # If Y greater than current X, then this is not a feasible policy, returning -infinite reward
   if (y > x):
    k["Reward"] = -math.inf
    return
    
   # Else, we use the given function to calculate reward
   k["Reward"] = k["Reward"] + g(y) + h(x-y)
  
   # Calculating new value for X
   x = alpha*y + beta*(x-y)
  
######## Configuring Korali Experiment

import korali
  
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = environment

# Defining problem's state.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Type"] = "State"
e["Variables"][0]["Parameter Space"] = "Discrete"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = initialX
e["Variables"][0]["Granularity"] = granularity

# Defining problem's actions.
e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Parameter Space"] = "Discrete"
e["Variables"][1]["Lower Bound"] = 0.0
e["Variables"][1]["Upper Bound"] = initialX
e["Variables"][1]["Granularity"] = granularity

# Configuring the solver
e["Solver"]["Type"] = "Learner/QTable"
e["Solver"]["Q Update Algorithm"] = "Q-Learning"
e["Solver"]["Initial Q Value"] = 50

# Configuring Output
e["Console Output"]["Frequency"] = 5000
e["File Output"]["Enabled"] = False

######## Running Korali and printing results

k = korali.Engine()
k.run(e)

print('Best Policy:     ' + str(e["Results"]["Optimal Policy Actions"]))
print('Optimal Reward:  ' + str(e["Results"]["Optimal Reward"]))
