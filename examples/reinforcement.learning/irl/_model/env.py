#!/usr/bin/env python3
import numpy as np
from cartpole import *

######## Defining Environment Storage

cart = CartPole()
maxSteps = 500

def env(s):

 # Initializing environment
 cart.reset()
 state = cart.getState().tolist()
 s["State"] = state
  
 # Getting Features
 # Pole angle and cart velocity^2
 s["Features"] = [ np.cos(state[2]), state[1]*state[1] ]
 # Pole angle, cart velocity^2 and dummy
 #s["Features"] = [ np.cos(state[2]), state[1]*state[1], np.random.normal(0.0, 0.1) ]
 
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
  s["State"] = state
  
  # Getting Features
  # Pole angle and cart velocity^2
  s["Features"] = [ np.cos(state[2]), state[1]*state[1] ]
  # Pole angle, cart velocity^2 and dummy
  #s["Features"] = [ np.cos(state[2]), state[1]*state[1], np.random.normal(0.0, 0.1) ]
  
  # Advancing step counter
  step = step + 1

 print(f"Steps {step}/{maxSteps}")
 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

def cosenv(s):

 # Initializing environment
 cart.reset()
 state = cart.getState().tolist()
 s["State"] = state
 #s["Features"] = [ state[2], np.random.normal(0.0, 1.0) ]
 s["Features"] = [ state[2] ]
 
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  #print(s["Action"]) 
  
  # Getting Reward
  s["Reward"] = 0.0
  
  # Storing New State
  state = cart.getState().tolist()
  s["State"] = cart.getState().tolist()
  
  # Getting Features
  #s["Features"] = [ state[2], np.random.normal(0.0, 1.0) ]
  s["Features"] = [ state[2] ]
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
