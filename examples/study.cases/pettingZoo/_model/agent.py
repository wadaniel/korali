#!/usr/bin/env python3

import math
import pdb
import numpy as np
import os
from PIL import Image

def initEnvironment(e, envName, moviePath = ''):

 # Creating environment 
 if (envName ==  'Waterworld'):
    from pettingzoo.sisl import waterworld_v3
    
    
    env = waterworld_v3.env()

    stateVariableCount = 242
    actionVariableCount = 2
    eps = 10 ** (-20)
    obs_upper = 2 * math.sqrt(2)
    obs_low = -1 * math.sqrt(2)
    ac_upper = 0.01 - eps
    ac_low = -0.01 + eps
    numIndividuals = 5
 
 
 
 ### Defining problem configuration for openAI Gym environments
 e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
 e["Problem"]["Environment Function"] = lambda x : agent(x, env)
 e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
 e["Problem"]["Training Reward Threshold"] = math.inf
 e["Problem"]["Testing Frequency"] = 201
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
  
 # Defining Action Variables
 
 for i in range(actionVariableCount):
  e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
  e["Variables"][stateVariableCount + i]["Type"] = "Action"
  e["Variables"][stateVariableCount + i]["Lower Bound"] = float(ac_low)
  e["Variables"][stateVariableCount + i]["Upper Bound"] = float(ac_upper)
  e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = math.sqrt(0.2)
 
 ### Defining Termination Criteria

 e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = math.inf
 

def agent(s, env):

 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False
 
 env.reset()
 
 states = []
 
 for ag in env.agents:
  state = env.observe(ag).tolist()
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
      obs=env.env.env.env.render('rgb_array')
      im = Image.fromarray(obs)
      fname = os.path.join("/scratch/mzeqiri/korali/examples/study.cases/pettingZoo/images/","image_{0}.png".format(image_count))
      im.save(fname)
      image_count += 1
   observation, reward, done, info = env.last()
   rewards.append(reward)
   action = actions.pop(0)   
   env.step(np.array(action,dtype= 'float32'))
   


  # Getting Reward
  s["Reward"] = rewards
  
  # Storing New State
  states = []
 
  for ag in env.agents:
   state = env.observe(ag).tolist()
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
