#!/usr/bin/env python3
import numpy as np

# Problem taken from https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Optimal-Control/Recitations/recitation_day_3.pdf

######## Defining Problem's Constants

N = 2 # Number of Ovens
minTemp = 0.0   # Min Oven T
maxTemp = 300.0 # Max Oven T
t0 = 50 # Initial Object Temperature
T = 200 # Target Object Temperature
r = 5.0 # Tdiff Penalization Multiplier
alpha = 0.6 # Heat Conductivity
intervals = 100 # How fine will we discretize the variable space

######## Defining Problem's Formulae

# Oven Temperature Penalization Formula
def penalizationFormula(u):  
 return u*u

# Heating Formula
def heatingFormula(x, u):
 return (1.0-alpha)*x + alpha*u

# Environment that provides the state/action loop
def environment(k):

  # Setting initial object temperature
  t = t0 

  # Variable to store the cummulative costs of using the ovens
  penalizationSum = 0 

  # Setting initial state (object temperature)
  k["State"] = [ t ]
   
  # Run the policy for N stages
  for i in range(N):

   # Get back to Korali to obtain the next action to perform
   k.update()
   
   # Obtaining oven temperature from action
   u = k["Action"][0]
    
   # Adding penalization for using the oven at the specified temperature
   penalizationSum = penalizationSum + penalizationFormula(u)
    
   # Calculate new object temperature
   t = heatingFormula(t,u)
     
   # Evaluating reward model
   k["Reward"] = - r*(t-T)*(t-T) - penalizationSum

   # Setting new State (object temperature)
   k["State"] = [ t ]
   
######## Configuring Korali Experiment

import korali

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Reinforcement Learning"
e["Problem"]["Environment Function"] = environment

# Defining problem's state.
e["Variables"][0]["Name"] = "Object Temperature"
e["Variables"][0]["Type"] = "State"
e["Variables"][0]["Parameter Vector"] = np.linspace(minTemp, maxTemp, intervals, True).tolist()

# Defining problem's actions.
e["Variables"][1]["Name"] = "Oven Temperatures"
e["Variables"][1]["Type"] = "Action"
e["Variables"][1]["Parameter Vector"] = np.linspace(minTemp, maxTemp, intervals, True).tolist()

# Configuring the solver
e["Solver"]["Type"] = "Learner/QTable"
e["Solver"]["Q Update Algorithm"] = "Q-Learning"
e["Solver"]["Initial Q Value"] = 0

# Configuring Output
e["Console Output"]["Frequency"] = 5000
e["File Output"]["Enabled"] = False

######## Running Korali and printing results

k = korali.Engine()
k.run(e)

print('Optimal Oven Temps:   ' + str(e["Results"]["Optimal Policy Actions"]))
print('Optimal Object Temps: ' + str(e["Results"]["Optimal Policy States"]))
print('Optimal Final Cost:   ' + str(-e["Results"]["Optimal Reward"]))
