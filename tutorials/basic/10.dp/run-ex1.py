#!/usr/bin/env python3

# Starting Korali's Engine
import korali
k = korali.Engine()

N = 2
alpha = 0.5
beta = 0.5

def g(x):
 return x*x

def h(x):
 return x*x

def model(k):
  y = k["Variables"][0]
  x = 1 - y
  k["Cost Evaluation"] = g(y) + h(x-y);
  
def constraint(k):
 k["Constraint Evaluation"] = 0
 for decision in k["Policy"]:
  x = decision[0]
  if (x > 0.8):
   k["Constraint Evaluation"] = -1
   return
  
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "DP"
e["Problem"]["Cost Function"] = model
e["Problem"]["Policy Constraints"] = [ constraint ]

# Defining the problem's variables.
e["Variables"][0]["Name"] = "Y"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0
e["Variables"][0]["Interval Count"] = 100

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Interpolator"
e["Solver"]["Interpolation Strategy"] = "Nearest Neighbor" 
e["Solver"]["Termination Criteria"]["Recursion Depth"] = 5

# Running Korali
k.run(e)

bestPolicyIdx = e["Results"]["Best Policy Index"] 
print(e["Results"]["Optimal Policies"][bestPolicyIdx])
