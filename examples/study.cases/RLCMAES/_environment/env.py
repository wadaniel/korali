#!/usr/bin/env python3
import os
from objective import *

######## Defining Environment Storage

maxSteps = 100

def env(s, populationSize):

 # Initializing environment
 objective = ObjectiveFactory(populationSize)
 
 outfile = "history.npz"
 #outfile = s["Custom Settings"]["Output"]
 #print(outfile)
 if s["Custom Settings"]["Evaluation"] == "True":
    objective.reset(noisy=False)
 else:
    objective.reset(noisy=True)

 s["State"] = objective.getState().tolist()
 step = 0
 done = False

 objectives = []
 scales = []

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = objective.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = objective.getReward()
   
  # Storing New State
  s["State"] = objective.getState().tolist()
  
  # Advancing step counter
  objectives.append(objective.curBestF)
  scales.append(objective.scale)

  step = step + 1

 print("Objective: {}".format(objective.name))
 print("Initial Ef {} -- Terminal Ef {}".format(objective.initialEf, objective.curEf))
 print("Initial Best F {} -- Terminal Best F {} -- Best Ever F {}".format(objective.initialBestF, objective.curBestF, objective.bestEver))
 print("Terminal Scale\n{}".format(objective.scale))
 print("Terminal Mean\n{}".format(objective.mean))
 print("Terminal Cov\n{}".format(objective.cov))
 # Setting finalization status
 if (objective.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
 
 # Store statistics
 if s["Custom Settings"]["Evaluation"] == "True":
    # load previous
    if os.path.isfile(outfile):
        history = np.load(outfile)
        scaleHistory = history['scaleHistory']
        objectiveHistory = history['objectiveHistory']
        print(scaleHistory)
        scaleHistory = np.concatenate((scaleHistory, [scales]))
        objectiveHistory = np.concatenate((objectiveHistory, [objectives]))
    else:
        scaleHistory = [scales]
        objectiveHistory = [objectives]
        print(scales)
     
    np.savez(outfile, scaleHistory=scaleHistory, objectiveHistory=objectiveHistory)
