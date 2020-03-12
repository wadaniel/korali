#!/usr/bin/env python3
import math

# Starting Korali's Engine
import korali
k = korali.Engine()

# Number of Ovens
N = 2

# Oven Temperature Thresholds
minTemp = 0.0
maxTemp = 300.0

# Target Temperature
T = 200

# Initial Temperature
t0 = 0

# Temperature Difference Penalization Multiplier
r = 5.0

# Oven Temperature Penalization Formula
def penalization(u):
 return u*u

# Heating Formula
alpha = 0.6
def heating(x, u):
 return (1.0-alpha)*x + alpha*u

def rewardFunction(k):
  # Determining current recursion depth
  i = k["Current Depth"]
  
  # Determining final temperature based on policy
  penalizationSum = 0
  t = t0
  
  for p in k["Policy"]:
    # Obtaining oven temperature from policy
    u = p[0]
    
    # Adding penalization
    penalizationSum = penalizationSum + penalization(u)
    
    # Calculate new temperature
    t = heating(t,u)
     
  # The constraints are satisfied, evaluate reward model
  k["Cost Evaluation"] = r*(t-T)*(t-T) + penalizationSum
  
# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "DynamicProgramming"
e["Problem"]["Cost Function"] = rewardFunction

# Defining the problem's variables.
e["Variables"][0]["Name"] = "U"
e["Variables"][0]["Lower Bound"] = minTemp
e["Variables"][0]["Upper Bound"] = maxTemp
e["Variables"][0]["Interval Count"] = 300

# Configuring the discretizer solver's parameters
e["Solver"]["Type"] = "RecursiveDiscretizer"
e["Solver"]["Termination Criteria"]["Recursion Depth"] = N

# Running Korali
k.run(e)

print('Qmax: ' + str(e["Results"]["Optimal Policy"]))
print('F(Qmax) = ' + str(-e["Results"]["Policy Evaluation"]))

t = t0
for p in e["Results"]["Optimal Policy"]:
 u = p[0]
 t = heating(t,u)
print('Final Temperature: ' + str(t))