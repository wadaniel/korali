#!/usr/bin/env python3
from objective import *

######## Defining Environment Storage

maxSteps = 100

def env(s, populationSize):

 # Initializing environment
 objective = ObjectiveFactory(populationSize)
 objective.reset()
 s["State"] = objective.getState().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = objective.advance(s["Action"])
  #print(s["Action"]) 
  
  # Getting Reward
  s["Reward"] = objective.getReward()
   
  # Storing New State
  s["State"] = objective.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 print("Initial Ef {} -- Terminal Ef {}".format(objective.initialEf, objective.curEf))
 print("Initial Best F {} -- Terminal Best F {} -- Best Ever F {}".format(objective.initialBestF, objective.curBestF, objective.bestEver))
 print("Terminal Mean {}".format(objective.mean))
 print("Terminal Scale {}".format(objective.scale))
 # Setting finalization status
 if (objective.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
