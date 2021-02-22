#!/usr/bin/env python3
import gym

def env(s):

 # Creating Environment object
 maxSteps = 1000
 cart = gym.make('CartPole-v1').unwrapped
 
 # Checking whether to save a movie
 saveMovie = False
 if (s["Custom Settings"]["Save Movie"] == "Enabled"):
  saveMovie = True
  cart = gym.wrappers.Monitor(cart, s["Custom Settings"]["Movie Path"], force=True)
 
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

  # Printing movie step generation    
  if (saveMovie):  print('[Korali] Frame ' + str(step), end = '')
    
  # Performing the action
  state, reward, done, info = cart.step(action)

  # Storing Reward
  s["Reward"] = reward

  # Printing movie information
  if (saveMovie):  print(' - State: ' + str(state) + ' - Action: ' + str(action))
       
  # Storing New State
  s["State"] = state.tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting termination status
 if (done):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"