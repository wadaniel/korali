#!/usr/bin/env python3
import csv
from cartpole import *

######## Defining Environment Storage

maxSteps = 500

def env(s, th):

 salist = []

 # Initializing environment
 cart = CartPole(th)
 
 s["State"] = cart.getState().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = cart.getReward()
   
  # Storing New State
  state = cart.getState().tolist()
  s["State"] = state
  
  if s["Mode"] == "Testing" and s["Custom Settings"]["Record Observations"] == True:
      stateaction = s["State"]
      action = s["Action"]
      stateaction.append(action[0])
      salist.append(stateaction)
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

 if s["Mode"] == "Testing" and s["Custom Settings"]["Record Observations"] == True:
    print("Generating observations for sample {0}".format(s["Sample Id"]))
    print("Observations recorded: {0}".format(len(salist)))
    print("Reward during recoding: {0}".format(s["Reward"]))

    with open('observations.csv', 'a') as myfile:
      wr = csv.writer(myfile)
      for stateaction in salist:
        wr.writerow(stateaction)
