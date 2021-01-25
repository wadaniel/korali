#!/usr/bin/env python3
from swimmer import *

######## Defining Environment Storage

swimmer = SwimmerEnv()
maxSteps = 100

def env(s):

 # Initializing environment
 swimmer.reset_model()
 s["State"] = swimmer._get_obs().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  print(s["Action"])
  state, reward, done, _ = swimmer.step(s["Action"])
  
  # Getting Reward
  s["Reward"] = reward
   
  # Storing New State
  s["State"] = state.tolist()
   
  # Advancing step counter
  step = step + 1

 if (done):
   s["Termination"] = "Normal"
 else:
   s["Termination"] = "Truncated"
 
if __name__ == "__main__":
  print("Start Main..")
  print("lower / upper action1 space of env: {}, {}". format(swimmer.action_space.low[0], swimmer.action_space.high[0]))
  print("lower / upper action2 space of env: {}, {}". format(swimmer.action_space.low[1], swimmer.action_space.high[1]))
  print("Reset..")
  swimmer.reset_model()
  state = swimmer._get_obs()
  print("Get State (len {}) ..".format(len(state)))
  print(state)
  print("Apply Force [+0.5, +0.5] ..")
  action = [+5.0, 5.0]
  state, reward, done, _ = swimmer.step(action)
  print("New State (len {}) ..".format(len(state)))
  print(state)
  print("Reward..")
  print(reward)
  print(done)
  print("Done!")
  
