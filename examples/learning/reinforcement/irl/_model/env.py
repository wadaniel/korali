#!/usr/bin/env python3
from cartpole import *

######## Defining Environment Storage

cart = CartPole()
maxSteps = 500

def env(s):

 # Initializing environment
 cart.reset()
 s["State"] = cart.getState().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  #print(s["Action"]) 
  
  # Getting Reward
  s["Reward"] = cart.getReward()
  
  # Storing New State
  state = cart.getState().tolist()
  s["State"] = cart.getState().tolist()
  
  # Getting Features
  #s["Features"] = [ np.cos(state[2]) ]
  s["Features"] = [ np.cos(state[2]), state[1]*state[1] ]
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
