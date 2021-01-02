#!/usr/bin/env python3
from lander import *

######## Defining Environment Storage

lander = Lander()

def env(s):

 cumulativeReward = 0
 
 # Initializing environment
 sampleId = s["Sample Id"]
 lander.reset(sampleId)
 s["State"] = lander.getState().tolist()
 step = 0
 done = False
 
 maxSteps = 200
 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = lander.advance(s["Action"])
  #print(s["Action"]) 
  
  # Getting Reward
  reward = lander.getReward()
  cumulativeReward = cumulativeReward + reward
  s["Reward"] = reward
   
  # Storing New State
  s["State"] = lander.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 #print(str(lander.getState()), end="")
 #print(" Step: " + str(step), end="")
 #print(" Reward: " + str(cumulativeReward))

 # Setting finalization status
 if (lander.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

  