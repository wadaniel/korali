#!/usr/bin/env python3
import gym

######## Defining Environment Storage

cart = gym.make('CartPole-v1').unwrapped
maxSteps = 1000

def env(s):

 # Initializing environment
 seed = s["Sample Id"]
 cart.seed(seed)
 
 # Resetting and getting initial state
 s["State"] = cart.reset().tolist()
 
 # Setting termination conditions
 step = 0
 done = False

 # Running steps until termination is met
 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Reading action
  action = s["Action"][0] 
    
  # Performing the action
  state, reward, done, info = cart.step(action)

  # Storing Reward
  s["Reward"] = reward
   
  # Storing New State
  s["State"] = state.tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting termination status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"