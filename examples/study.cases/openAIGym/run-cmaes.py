#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

import sys
import math
import numpy as np

sys.path.append('_model')
import gym
#import pyBulletEnvironments
from HumanoidWrapper import HumanoidWrapper
from AntWrapper import AntWrapper

# Define Parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)

def agent(s, env):
 
 print("X")
 stateVariableCount = env.observation_space.shape[0]
 actionVariableCount = env.action_space.shape[0]
 
 # Get policy
 X = p["Parameters"]
 X.reshape((stateVariableCount, actionVariableCount))
 
 state = env.reset()
 step = 0
 done = False
 
 # Storage for cumulative reward
 cumulativeReward = 0.0
 overSteps = 0
 while not done and step < 1000:
  # Getting new action
  s.update()
  
  # Performing the action
  action = np.dot(state, X)
  state, reward, done, _ = env.step(action)
  
  # Update cumulative reward
  cumulativeReward = cumulativeReward + reward 
  
  # Advancing step counter
  step = step + 1
 
  s["Objective"] = cumulativeReward

if __name__ == '__main__':
  
  args = parser.parse_args()

  envName = args.env
  resultFolder = '_result_vracer_' + envName + '/'
 
  # Environment specifics
  env = gym.make(envName)
  if (envName == 'Humanoid-v2'):
    env = HumanoidWrapper(env)
  if (envName == 'HumanoidStandup-v2'):
    env = HumanoidWrapper(env)
  if (envName == 'Ant-v2'):
    env = AntWrapper(env)
   
  stateVariableCount = env.observation_space.shape[0]
  actionVariableCount = env.action_space.shape[0]
   
  stateVariablesIndexes = range(stateVariableCount)
  if (envName == 'Ant-v2'):
    stateVariableCount = 27
    stateVariablesIndexes = range(stateVariableCount) 


  # Starting Korali's Engine
  import korali
  k = korali.Engine()
  e = korali.Experiment()

  #e.loadState(resultFolder + '/latest');

  e["Random Seed"] = 0xC0FEE
  e["Problem"]["Type"] = "Optimization"
  e["Problem"]["Objective Function"] = lambda s : agent(s, env)

  dim = stateVariableCount * actionVariableCount

  # Defining the problem's variables.
  for i in range(dim):
      e["Variables"][i]["Name"] = "X" + str(i)
      e["Variables"][i]["Lower Bound"] = -5.0
      e["Variables"][i]["Upper Bound"] = +5.0
      e["Variables"][i]["Initial Standard Deviation"] = 1e-2

  # Configuring CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/CMAES"
  e["Solver"]["Population Size"] = 8
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-32
  e["Solver"]["Termination Criteria"]["Max Generations"] = 1000

  # Configuring results path
  e["File Output"]["Enabled"] = True
  e["File Output"]["Path"] = resultFolder
  e["File Output"]["Frequency"] = 10

  # Running Korali
  k.run(e)
