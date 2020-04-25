#!/usr/bin/env python3

# Problem taken from https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Optimal-Control/Recitations/recitation_day_3.pdf

######## Defining Problem's Constants

N = 2 # Number of Ovens
minTemp = 0.0   # Min Oven T
maxTemp = 300.0 # Max Oven T
t0 = 0 # Initial Temperature
T = 200 # Target Temperature
r = 5.0 # Tdiff Penalization Multiplier
alpha = 0.6 # Heat Conductivity
intervals = 200 # How fine will we discretize the variable space

######## Defining Problem's Formulae

# Oven Temperature Penalization Formula
def penalization(u):  
 return u*u

# Heating Formula
def heating(x, u):
 return (1.0-alpha)*x + alpha*u

# Cost function to optimize
def costFunction(k):

  penalizationSum = 0 # Variable to store the cummulative oven costs
  t = t0 # Variable to store the current temperature

  # Iterating the policy choices to determine final temperature and costs  
  for p in k["Policy"]:
    # Obtaining oven temperature from policy
    u = p[0]
    
    # Adding penalization
    penalizationSum = penalizationSum + penalization(u)
    
    # Calculate new temperature
    t = heating(t,u)
     
  # Evaluating cost model
  k["Cost Evaluation"] = r*(t-T)*(t-T) + penalizationSum

######## Configuring Korali Experiment

import korali

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Optimal Control"
e["Problem"]["Cost Function"] = costFunction

# Defining the problem's variables to discretize.
e["Variables"][0]["Name"] = "U"
e["Variables"][0]["Lower Bound"] = minTemp
e["Variables"][0]["Upper Bound"] = maxTemp
e["Variables"][0]["Interval Count"] = intervals

# Configuring the discretizer solver's parameters
e["Solver"]["Type"] = "Dijkstra"
e["Solver"]["Termination Criteria"]["Recursion Depth"] = N

######## Running Korali and printing results

k = korali.Engine()
k.run(e)

print('Best Policy:  ' + str(e["Results"]["Optimal Policy"]))
print('Optimal Cost: ' + str(e["Results"]["Policy Evaluation"]))

t = t0
for p in e["Results"]["Optimal Policy"]:
 u = p[0]
 t = heating(t,u)
print('Final Temperature: ' + str(t))
