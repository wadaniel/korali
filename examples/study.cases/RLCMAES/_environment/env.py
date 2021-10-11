#!/usr/bin/env python3
import os
from objective import *

######## Defining Environment Storage

maxSteps = 100

#objectiveList = ["fsphere", "felli", "fcigar", "ftablet", "fcigtab", "ftwoax", "fdiffpow", "rosenbrock", "fparabr", "fsharpr"]
objectiveList = ["fsphere", "felli", "fparabr", "booth", "rosenbrock", "dixon", "ackley", "levi", "rastrigin" ]
 
def env(s, objective, dim, populationSize, noise, version):

 # Selecting environment
 if objective == "random":
     envId = s["Sample Id"] % len(objectiveList)
     s["Environment Id"] = envId
     objective = objectiveList[envId]

 # Initializing environment
 objective = ObjectiveFactory(objective, dim, populationSize, version)
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

 # Setting finalization status
 if (objective.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
 
 # Store statistics
 if s["Custom Settings"]["Evaluation"] == "True":
    # load previous
    outfile = s["Custom Settings"]["Outfile"]
    outfile = outfile.replace("random", objective.objective)
    if os.path.isfile(outfile):
        history = np.load(outfile)
        scaleHistory = history['scaleHistory']
        objectiveHistory = history['objectiveHistory']
        muobjectiveHistory = history['muobjectiveHistory']
        actionHistory = history['actionHistory']

        scaleHistory = np.concatenate((scaleHistory, [scales]))
        objectiveHistory = np.concatenate((objectiveHistory, [objectives]))
        muobjectiveHistory = np.concatenate((objectiveHistory, [muobjectives]))
        actionHistory = np.concatenate((actionHistory, [actions]))

    else:
        scaleHistory = [scales]
        objectiveHistory = [objectives]
        muobjectiveHistory = [muobjectives]
        actionHistory = [actions]
     
    np.savez(outfile, scaleHistory=scaleHistory, objectiveHistory=objectiveHistory, muobjectiveHistory=muobjectiveHistory, actionHistory=actionHistory)
