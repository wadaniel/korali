#!/usr/bin/env python3

import gym
import pyBulletEnvironments
import math


def initEnvironment(e, envName, moviePath = ''):

 # Checking whether to save a movie
 
 if (moviePath != ''):
  env = gym.make(envName).unwrapped
  env = gym.wrappers.Monitor(env, moviePath, force=True)
 else:
  env = gym.make(envName)
 
 ### Defining problem configuration for openAI Gym environments
 e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
 e["Problem"]["Environment Function"] = lambda x : agent(x, env)
 e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
 e["Problem"]["Training Reward Threshold"] = math.inf
 e["Problem"]["Policy Testing Episodes"] = 20
 
 # Getting environment variable counts
 stateVariableCount = env.observation_space.shape[0]
 actionVariableCount = env.action_space.shape[0]
 
 # Defining State Variables
 
 for i in range(stateVariableCount):
  e["Variables"][i]["Name"] = "State Variable " + str(i)
  e["Variables"][i]["Type"] = "State"
  e["Variables"][i]["Lower Bound"] = float(env.observation_space.low[i])
  e["Variables"][i]["Upper Bound"] = float(env.observation_space.high[i])
  
 # Defining Action Variables
 
 for i in range(actionVariableCount):
  e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i)
  e["Variables"][stateVariableCount + i]["Type"] = "Action"
  e["Variables"][stateVariableCount + i]["Lower Bound"] = float(env.action_space.low[i])
  e["Variables"][stateVariableCount + i]["Upper Bound"] = float(env.action_space.high[i])
  e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = 0.25

 ### Defining Termination Criteria

 e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = math.inf
 
def agent(s, env):

 if (s["Custom Settings"]["Print Step Information"] == "Enabled"):
  printStep = True
 else:
  printStep = False
  
 s["State"] = env.reset().tolist()
 step = 0
 done = False

 # Storage for cumulative reward
 cumulativeReward = 0.0
  
 while not done and step < 1000:

  # Getting new action
  s.update()
  
  # Printing step information    
  if (printStep):  print('[Korali] Frame ' + str(step), end = '')
  
  # Performing the action
  action = s["Action"]
  state, reward, done, _ = env.step(action)
  
  # Getting Reward
  s["Reward"] = reward
  
  # Printing step information
  #if (printStep):  print(' - State: ' + str(state) + ' - Action: ' + str(action))
  cumulativeReward = cumulativeReward + reward 
  if (printStep):  print(' - Cumulative Reward: ' + str(cumulativeReward))
   
  # Storing New State
  s["State"] = state.tolist()
   
  # Advancing step counter
  step = step + 1

 # Setting termination status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
 
