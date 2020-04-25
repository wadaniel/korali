#!/usr/bin/env python3
import math

######## Defining Problem's Configuration

budget = 1 # Initial Budget (x)
intervals = 100 # How fine will we discretize the variable space

# Defining list of cost functions, one per stage 
evalFunctions = [
  lambda x : 2*x,
  lambda x : -30*x*x + 17.63*x
  ]

# The number of stages equals the number of functions defined above
N = len(evalFunctions)

# Reward function to optimize

def rewardFunction(k):
  
  # Adding up all the policy decisions to see if it exceeds budget
  sum = 0
  for decision in k["Policy"]:
   x = decision[0]
   sum = sum + x

  # Obtaining current recursion depth to check constraints
  i = k["Current Depth"]
  
  # If this is not the last stage, we check if we have not exceeded the budget so far
  if (i < N-1):
   if (sum >  budget): 
    # If we exceeded the budget, then this policy is non-viable. We set cost evaluation to infinity, 
    # so that Korali will not select this policy.
    k["Cost Evaluation"] = math.inf
    return
    
  # If this is the last stage, we check whether this last choice fulfills the budget (no more, no less).
  else: 
   if (sum != budget): 
    # If there is money left in the budget, this is not a viable policy. Returning infinity cost
    k["Cost Evaluation"] = math.inf
    return 
    
  # Since Korali minimizes cost, we maximize the reward by negating it
  k["Cost Evaluation"] = -evalFunctions[i](x)

######## Configuring Korali Experiment

import korali

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Optimal Control"
e["Problem"]["Cost Function"] = rewardFunction

# Defining problem variables to discretize.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = budget 
e["Variables"][0]["Interval Count"] = intervals

# Configuring the discretizer solver's parameters
e["Solver"]["Type"] = "Dijkstra"
e["Solver"]["Termination Criteria"]["Recursion Depth"] = N

######## Running Korali and printing results

k = korali.Engine()
k.run(e)

print('Qmax: ' + str(e["Results"]["Optimal Policy"]))
print('F(Qmax) = ' + str(-e["Results"]["Policy Evaluation"]))
