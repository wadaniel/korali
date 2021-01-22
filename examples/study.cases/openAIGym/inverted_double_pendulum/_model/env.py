#!/usr/bin/env python3

import gym

######## Defining Environment Storage

pendulum = gym.make('InvertedDoublePendulum-v2').unwrapped
maxSteps = 100

def env(s):

 # Initializing environment
 pendulum.reset_model()
 s["State"] = pendulum._get_obs().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  #print(s["Action"]) 
  state, reward, done, _ = pendulum.step(s["Action"])
  
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
 
if __name__ == "__main__":
  print("Start Main..")
  print("lower / upper action space of env: {}, {}". format(pendulum.action_space.high[0], pendulum.action_space.low[0]))
  print("Reset..")
  pendulum.reset_model()
  print("Get Obs..")
  state = pendulum._get_obs()
  print(state)
  print("Apply Force [+5.0] ..")
  action = [+5.0]
  state, reward, done, _ = pendulum.step(action)
  print("New Obs..")
  print(state)
  print("Reward..")
  print(done)
  print("Done!")
  
