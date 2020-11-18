#!/usr/bin/env python3
from abf import *

######## Defining Environment Storage

maxReward = -100000
swimmers = Swimmers()

def env(s):

 # Initializing environment
 swimmers.reset()
 
 s["State"] = swimmers.getState().tolist()
 
 cumulativeReward = 0
 done = False
 while not done:

  # Getting new action
  s.update()
  
  # Performing the action
  done = swimmers.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = swimmers.getReward()
   
  # Storing New State
  s["State"] = swimmers.getState().tolist()

  cumulativeReward = cumulativeReward + s["Reward"]
  
 # Setting finalization status
 if (swimmers.isSuccess()):
  s["Termination"] = "Normal"
 else:
  s["Termination"] = "Truncated"
  
 # Saving the trajectory for visualization
 global maxReward
 if (cumulativeReward > maxReward):
  maxReward = cumulativeReward
  swimmers.dumpTrajectoryToCsv('best.csv')
 swimmers.dumpTrajectoryToCsv('last.csv')