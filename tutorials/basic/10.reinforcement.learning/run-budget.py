#!/usr/bin/env python3
import math
import numpy as np

######## Defining Problem's Configuration

initialBudget = 1.0 # Initial Budget (x)
granularity = 0.005 # Discretization factor: how fine will we discretize the variable space
N = 2 # Number of stages

# Defining list of cost functions, one per stage 
evalFunctions = [
  lambda x : 2*x,
  lambda x : -30*x*x + 17.63*x
  ]

# Environment that provides the state/action loop
def environment(k):

 # Getting budget, current step, and amount to spend.
 budget     = k["State"][0]
 step       = int(k["State"][1])
 spenditure = k["Action"][0]
  
 # If Y greater than current X, then this is not a feasible policy, returning -infinite reward
 if (spenditure > budget): spenditure = budget
    
 # We use the given function to calculate reward
 k["Reward"] = evalFunctions[step](spenditure)
  
 newStep = float(step + 1)
 newBudget = budget - spenditure
 
 # Calculating new state (remaining budget)
 k["State"] = [ newBudget, newStep ]

######## Configuring Korali Experiment

import korali

# Instantiating Engine
k = korali.Engine()

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

e["Variables"][1]["Name"] = "Step"
e["Variables"][1]["Type"] = "State"
e["Variables"][1]["Parameter Space"] = "Custom"
e["Variables"][1]["Parameter Vector"] = [ 0.0, 1.0 ]

# Defining problem's actions.
e["Variables"][2]["Name"] = "Spenditure"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Parameter Space"] = "Discrete"
e["Variables"][2]["Lower Bound"] = 0.0
e["Variables"][2]["Upper Bound"] = initialBudget
e["Variables"][2]["Granularity"] = granularity

# Configuring the solver
e["Solver"]["Type"] = "Dynamic Programming"
e["Solver"]["Recursion Depth"] = N

# Silencing Korali's Output
e["Console Output"]["Verbosity"] = "Silent"
e["File Output"]["Enabled"] = False

######## Performing Training Stage

e["Problem"]["Operation"] = "Training"
k.run(e)

######## Performing Running Stage

e["Problem"]["Operation"] = "Running"
e["Problem"]["Initial State"] = [ initialBudget, 0 ]
k.run(e)

print('Best Policy States:     ' + str(e["Results"]["Optimal Policy States"]))
print('Best Policy Actions:    ' + str(e["Results"]["Optimal Policy Actions"]))
print('Optimal Reward:  ' + str(e["Results"]["Optimal Policy Reward"]))