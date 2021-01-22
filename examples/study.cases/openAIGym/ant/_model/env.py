#!/usr/bin/env python3

import gym

######## Defining Environment Storage

ant = gym.make('Ant-v2').unwrapped
maxSteps = 100

def env(s):

 # Initializing environment
 ant.reset_model()
 print(ant.observation_space)
 print(ant.action_space.low)
 exit(0)
 s["State"] = ant._get_obs().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  #print(s["Action"]) 
  state, reward, done, _ = ant.step(s["Action"])
  
  # Getting Reward
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
 
