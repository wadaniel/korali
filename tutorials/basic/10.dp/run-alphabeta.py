#!/usr/bin/env python3
import math

# Starting Korali's Engine
import korali
k = korali.Engine()

N = 5
initialX = 1.0
alpha = 0.6
beta = 0.3

def g(x):
 return x*x

def h(x):
 return x*x*x

def rewardFunction(k):
  # Determining current recursion depth
  i = k["Current Depth"]
  
  # Recovering the value of x, based on constraints
  currentX = initialX
  
  for p in k["Policy"]:
    y = p[0]
    
    # Check if the value of y is valid
    if (y < 0):
     k["Cost Evaluation"] = math.inf
     return
    
    # Check if the value of y is valid 
    if (y > currentX):
     k["Cost Evaluation"] = math.inf
     return 
     
    # Updating the value of x, based on the selection
    x = currentX
    currentX = alpha*y + beta*(x-y)
     
  # The constraints are satisfied, evaluate reward model
  k["Cost Evaluation"] = - ( g(y) + h(x-y) )
  
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "DynamicProgramming"
e["Problem"]["Cost Function"] = rewardFunction

# Defining the problem's variables.
e["Variables"][0]["Name"] = "Y"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = initialX
e["Variables"][0]["Interval Count"] = 100

# Configuring the discretizer solver's parameters
e["Solver"]["Type"] = "RecursiveDiscretizer"
e["Solver"]["Termination Criteria"]["Recursion Depth"] = N

# Running Korali
k.run(e)

print('Qmax: ' + str(e["Results"]["Optimal Policy"]))
print('F(Qmax) = ' + str(-e["Results"]["Policy Evaluation"]))
