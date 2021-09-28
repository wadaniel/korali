#!/usr/bin/env python3

import math
import pdb
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def initEnvironment(e, envName, model = ''):

 # Creating environment 
 if (envName ==  'Pursuit'):
    from pettingzoo.sisl import pursuit_v3
    
    env = pursuit_v3.env()
    stateVariableCount = 147
    actionVariableCount = 1
    
    obs_upper = 30
    obs_low = 0
    numIndividuals = 8

 else:
    print("Environment '{}' not recognized! Exit..".format(envName))
    sys.exit()
 
 
 
 ### Defining problem configuration for openAI Gym environments
 e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
 e["Problem"]["Environment Function"] = lambda x : agent(x, env, model)
 e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
 e["Problem"]["Training Reward Threshold"] = math.inf
 e["Problem"]["Possible Actions"] = [ [0], [1], [2], [3], [4] ]
 #e["Problem"]["Testing Frequency"] = 2
 e["Problem"]["Policy Testing Episodes"] = 20
 e["Problem"]["Agents Per Environment"] = numIndividuals
 
 # Generating state variable index list
 stateVariablesIndexes = range(stateVariableCount)
 
 # Defining State Variables
 
 for i in stateVariablesIndexes:
  e["Variables"][i]["Name"] = "State Variable " + str(i)
  e["Variables"][i]["Type"] = "State"
  e["Variables"][i]["Lower Bound"] = float(obs_low)
  e["Variables"][i]["Upper Bound"] = float(obs_upper)
 if model == '1' :
  e["Variables"][stateVariableCount ]["Name"] = "State Variable " + str(i)
  e["Variables"][stateVariableCount ]["Type"] = "State"
  stateVariableCount += 1
  
 # Defining Action Variables
 
 for i in range(actionVariableCount):
  e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
  e["Variables"][stateVariableCount + i]["Type"] = "Action"

 
 ### Defining Termination Criteria

 e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = math.inf
 

def agent(s, env, model = ''):


 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False
 
 env.reset()
 
 states = []
 
 for ag in env.agents:
  state = env.observe(ag)
  state = state.reshape(147)
  state = state.tolist()
  if model == '1':
    state.append(float(ag[-1]))
  states.append(state)
 s["State"] = states
 
 step = 0
 done = False

 # Storage for cumulative reward
 cumulativeReward = 0.0
 
 overSteps = 0
 if s["Mode"] == "Testing":
   image_count = 0

  
 while not done and step < 500:

  s.update()
  
  # Printing step information    
  if (printStep):  print('[Korali] Frame ' + str(step), end = '')
  
  
  actions = s["Action"]
  rewards = []
  
  for ag in env.agents:
   if s["Mode"] == "Testing":
      pdb.set_trace()
      obs=env.env.env.env.render('rgb_array')
      im = Image.fromarray(obs)
      fname = os.path.join("/scratch/mzeqiri/korali/examples/study.cases/pettingZoo/images/images_pursuit/","image_{0}.png".format(image_count))
      im.save(fname)
      image_count += 1
      
      
   observation, reward, done, info = env.last()
   rewards.append(reward)
   #pdb.set_trace()
   action = actions.pop(0)
   env.step(action[0])
   
   


  # Getting Reward
  s["Reward"] = rewards
  
  # Storing New State
  states = []
 
  for ag in env.agents:
   state = env.observe(ag)
   state = state.reshape(147)
   state = state.tolist()
   if model == '1':
    state.append(float(ag[-1]))
   states.append(state)
  s["State"] = states
   
  # Advancing step counter
  step = step + 1

 # Setting termination status
 if (not env.agents):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

 if s["Mode"] == "Testing":
   env.close()
