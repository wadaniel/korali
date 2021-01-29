#!/usr/bin/env python3
import csv
from cartpole import *

######## Defining Environment Storage

maxSteps = 500

def env(s, th):

 salist = []
 
 # Initializing environment
 cart = CartPole(th)
 cart.reset()

 state = cart.getState().tolist()
 s["State"] = state
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  action = s["Action"]
  done = cart.advance(action)
  
  # Getting Reward
  s["Reward"] = cart.getReward()
   
  # Storing New State
  newstate = cart.getState().tolist()
  s["State"] = newstate
  
  if s["Custom Settings"]["Record Observations"] == "True":
      stateactionstate = state
      stateactionstate.append(action[0])
      stateactionstate.append(newstate)

      salist.append(stateactionstate)

  state = newstate
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

 if s["Custom Settings"]["Record Observations"] == "True":
    obsfile = s["Custom Settings"]["Output"]
    print("Generating observations for sample {0}".format(s["Sample Id"]))
    print("Observations recorded: {0}".format(len(salist)))
    print("Reward during recoding: {0}".format(s["Reward"]))

    with open(obsfile, 'a') as myfile:
      wr = csv.writer(myfile)
      for stateaction in salist:
        wr.writerow(stateaction)


def envWithTestObs(s, th):

 if s["Mode"] == "Training":

     # Initializing environment
     cart = CartPole(th)
     cart.reset()

     s["State"] = cart.getState().tolist()
     step = 0
     done = False

     while not done and step < maxSteps:

      # Getting new action
      s.update()
      
      # Performing the action
      action = s["Action"]
      done = cart.advance(action)
      
      # Getting Reward
      s["Reward"] = cart.getReward()
       
      # Storing New State
      state = cart.getState().tolist()
      s["State"] = state
      
      # Advancing step counter
      step = step + 1

     # Setting finalization status
     if (cart.isOver()):
      s["Termination"] = "Terminal"
     else:
      s["Termination"] = "Truncated"

 else: # Testing

     obsfile = s["Custom Settings"]["Output"]
     states = []
     obsactions = []

     with open(obsfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            states.append(row[:4])
            obsactions.append(row[4])

     suml2 = 0.0

     for i, state in enumerate(states):

        s["State"] = state

        # Getting new action
        s.update()
     
        action = s["Action"]

        # Compare with observations
        l2error = np.linalg.norm(np.array(obsactions[i])-np.array(action))
        s["Reward"] = -l2error

     # Done
     s["Termination"] = "Terminal"
