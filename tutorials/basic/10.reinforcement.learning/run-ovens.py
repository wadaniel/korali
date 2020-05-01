#!/usr/bin/env python3
import numpy as np

# Problem taken from https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Optimal-Control/Recitations/recitation_day_3.pdf

######## Defining Problem's Constants

N = 2 # Number of Ovens
minTemp = 0.0   # Min Oven T
maxTemp = 500.0 # Max Oven T
t0 = 50 # Initial Object Temperature
T = 200 # Target Object Temperature
r = 70.0 # Tdiff Penalization Multiplier
alpha = 0.2 # Heat Conductivity
granularity = 1.0 # Discretization factor: how fine will we discretize the variable space

######## Defining Problem's Formulae

# Oven Temperature Penalization Formula
def penalizationFormula(u):  
 return u*u

# Heating Formula
def heatingFormula(x, u):
 return (1.0-alpha)*x + alpha*u

# Environment that provides the state/action loop
def environment(k):

 # Getting object and oven temperature
 t = k["State"][0]
 u = k["Action"][0]
 step = int(k["State"][1])
     
 # Calculating new temperature
 newTemp = heatingFormula(t,u)
      
 # If this is an intermediate step, the penalization is just the use of the oven
 if (step < N - 1): k["Reward"] = -penalizationFormula(u)
 
 # Else, if this is the last step, then penalization is the use of the oven and the difference in temperature between the goal
 else: k["Reward"] = - r*(newTemp-T)*(newTemp-T) - penalizationFormula(u)  
 
 # Calculating new step
 newStep = float(step + 1)
 
 # Setting state
 k["State"] = [ newTemp, newStep ]
   
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
e["Variables"][0]["Name"] = "Object Temperature"
e["Variables"][0]["Type"] = "State"
e["Variables"][0]["Parameter Space"] = "Discrete"
e["Variables"][0]["Lower Bound"] = minTemp
e["Variables"][0]["Upper Bound"] = maxTemp
e["Variables"][0]["Granularity"] = granularity

# Defining problem's state.
e["Variables"][1]["Name"] = "Step"
e["Variables"][1]["Type"] = "State"
e["Variables"][1]["Parameter Space"] = "Custom"
e["Variables"][1]["Parameter Vector"] = [ 0, 1, 2 ]

# Defining problem's actions.
e["Variables"][2]["Name"] = "Oven Temperatures"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Parameter Space"] = "Discrete"
e["Variables"][2]["Lower Bound"] = minTemp
e["Variables"][2]["Upper Bound"] = maxTemp
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
e["Problem"]["Initial State"] = [ t0, 0 ]
k.run(e)

print('Best Policy States:     ' + str(e["Results"]["Optimal Policy States"]))
print('Best Policy Actions:    ' + str(e["Results"]["Optimal Policy Actions"]))
print('Optimal Reward:  ' + str(e["Results"]["Optimal Policy Reward"]))
