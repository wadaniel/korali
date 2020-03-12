#!/usr/bin/env python3
import numpy as np

# Starting Korali's Engine
import korali
k = korali.Engine()

budget = 1

evalFunctions = [
  lambda x : 2*x,
  lambda x : -30*x*x + 17.63*x
  ]

def model(k):
  i = k["Current Depth"]
  x = k["Variables"][0]
  k["Cost Evaluation"] = -evalFunctions[i](x)
  
def constraint(k):
 k["Constraint Evaluation"] = 1
 sum = 0
 for decision in k["Policy"]:
  sum = sum + decision[0]
 
 if (sum > budget):
   k["Constraint Evaluation"] = -1
 
 if (k["Current Depth"] == 0):
  if (sum != budget):  
   k["Constraint Evaluation"] = -1 
  
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "DynamicProgramming"
e["Problem"]["Cost Function"] = model
e["Problem"]["Policy Constraints"] = [ constraint ]

# Defining the problem's variables.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0
e["Variables"][0]["Interval Count"] = 100

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "RecursiveDiscretizer"
e["Solver"]["Termination Criteria"]["Recursion Depth"] = len(evalFunctions)

# Running Korali
k.run(e)

bestPolicyIdx = e["Results"]["Best Policy Index"] 
print('Qmax: ' + str(e["Results"]["Optimal Policies"][bestPolicyIdx]))
print('F(Qmax) = ' + str(-e["Results"]["Policy Evaluations"][bestPolicyIdx]))

res = evalFunctions[0](0.74) + evalFunctions[1](0.26) 
print(res)