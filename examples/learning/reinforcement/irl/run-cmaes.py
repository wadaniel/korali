#!/usr/bin/env python3

# Importing computational model
import sys
import math
import json

import numpy as np

obsfile = "observations-t-0.0.json"
obsstates = []
obsactions = []
with open(obsfile, 'r') as infile:
 obsjson = json.load(infile)
 obsstates = obsjson["States"]
 obsactions = obsjson["Actions"]

# Reward function
def reward(s):

  x = s["Parameters"]
  ### Compute Feauters from states
  th0 = x[0]
  th1 = x[1]
  th2 = x[2]

  reward = 0.0
  N = len(obsstates)
  M = int(N*0.9)

  for trajectory in obsstates:
      for state in trajectory:
          # Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
          feature1 = np.cos(state[2])
          feature2 = state[1]*state[1]
          feature3 = np.random.normal(0.0, 0.1) # dummy
          reward += th0*feature1 + th1*feature2 + th2*feature3
  
  reward /= N

  sig = 1
  K = 1
  Z = 0.0
  for trajectory in obsstates:
    for i in range(K):
      rewardTraj = 0.0
      for state in trajectory:
        #rs1 = state[1]
        #rs2 = state[2]
        rs1 = np.random.normal(state[1],sig)
        rs2 = np.random.normal(state[2],sig)
        # Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
        feature1 = np.cos(rs2)
        feature2 = rs1*rs1
        feature3 = np.random.normal(0.0, 0.1) # dummy
        rewardTraj += th0*feature1 + th1*feature2 + th2*feature3
      
      Z += np.exp(rewardTraj)
  
  Z /= (N*K)
  s["F(x)"] = reward - np.log(Z)

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = reward

dim = 3

# Defining the problem's variables.
for i in range(dim):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Lower Bound"] = -1e-5
    e["Variables"][i]["Upper Bound"] = +1e-5
    e["Variables"][i]["Initial Standard Deviation"] = 0.3*1e-5

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-32
e["Solver"]["Termination Criteria"]["Max Generations"] = 250

# Configuring results path
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)
