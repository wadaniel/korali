#!/usr/bin/env python3
import os
from objective import *

######## Defining Environment Storage

maxSteps = 100

def env(s, objective, dim, populationSize, noise):

 # Initializing environment
 objective = ObjectiveFactory(objective, dim, populationSize)
 
 
 if s["Custom Settings"]["Evaluation"] == "True":
    objective.reset(noise=0.0)
 else:
    objective.reset(noise=noise)

 s["State"] = objective.getState().tolist()
 step = 0
 done = False

 objectives = []
 muobjectives = []
 scales = []
 actions = []

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
  muobjectives.append(objective.curEf)
  scales.append(objective.scale)
  actions.append(s["Action"])

  step = step + 1

 #print("Objective: {}".format(objective.name))
 #print("Initial Ef {} -- Terminal Ef {}".format(objective.initialEf, objective.curEf))
 #print("Initial Best F {} -- Terminal Best F {} -- Best Ever F {}".format(objective.initialBestF, objective.curBestF, objective.bestEver))
 #print("Terminal Scale\n{}".format(objective.scale))
 #print("Terminal Mean\n{}".format(objective.mean))
 #print("Terminal Cov\n{}".format(objective.cov))
 
 # Setting finalization status
 if (objective.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
 
 # Store statistics
 if s["Custom Settings"]["Evaluation"] == "True":
    # load previous
    #outfile = s["Custom Settings"]["Output"]
    outfile = "history_{}_{}_{}_{}.npz".format(objective.objective, dim, populationSize, noise)
    if os.path.isfile(outfile):
        history = np.load(outfile)
        scaleHistory = history['scaleHistory']
        objectiveHistory = history['objectiveHistory']
        muobjectiveHistory = history['muobjectiveHistory']
        #actionHistory = history['actionHistory']

        scaleHistory = np.concatenate((scaleHistory, [scales]))
        objectiveHistory = np.concatenate((objectiveHistory, [objectives]))
        muobjectiveHistory = np.concatenate((objectiveHistory, [muobjectives]))
        #actionHistory = np.concatenate((actionHistory, [actions]))

    else:
        scaleHistory = [scales]
        objectiveHistory = [objectives]
        muobjectiveHistory = [muobjectives]
        #actionHistory = [actions]
     
    #np.savez(outfile, scaleHistory=scaleHistory, objectiveHistory=objectiveHistory, muobjectiveHistory=muobjectiveHistory, actionHistory=actionHistory)
    np.savez(outfile, scaleHistory=scaleHistory, objectiveHistory=objectiveHistory, muobjectiveHistory=muobjectiveHistory)

