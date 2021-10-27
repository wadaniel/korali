#!/usr/bin/env python3
import os
from objective import *

######## Defining Environment Storage


trainingObjectiveList = ["fsphere", "felli", "fcigar", "ftablet", "fcigtab", "ftwoax", "fdiffpow", "rosenbrock", "fparabr", "fsharpr"]
evaluationObjectiveList = ["fsphere", "felli", "fcigar", "ftablet", "fcigtab", "ftwoax", "fdiffpow", "rosenbrock", "fparabr", "fsharpr", "booth", "dixon", "ackley", "levi", "rastrigin"]
 
def env(s, objective, dim, populationSize, steps, noise, version):

 maxSteps = steps
 
 # Selecting environment
 if objective == "random":
 
    objectiveList = None
    if s["Custom Settings"]["Evaluation"] == "True":
        objectiveList = evaluationObjectiveList
    else:
        objectiveList = trainingObjectiveList

    envId = s["Sample Id"] % len(objectiveList)
    s["Environment Id"] = envId
    objective = objectiveList[envId]

 # Initializing environment
 objectiveFactory = ObjectiveFactory(objective, dim, populationSize, version)
 objectiveFactory.reset(noise=noise)

 s["State"] = objectiveFactory.getState().tolist()
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
  done = objectiveFactory.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = objectiveFactory.getReward()
   
  # Storing New State
  s["State"] = objectiveFactory.getState().tolist()
  
  #state = objectiveFactory.getState().tolist()
  #print(state)

  # Advancing step counter
  objectives.append(objectiveFactory.curBestF)
  muobjectives.append(objectiveFactory.curEf)
  scales.append(objectiveFactory.scale)
  actions.append(s["Action"])

  step = step + 1

 # Setting finalization status
 if (objectiveFactory.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
 
 # Store statistics
 if s["Custom Settings"]["Evaluation"] == "True":
    # load previous
    outfile = s["Custom Settings"]["Outfile"]
    outfile = outfile.replace("random", objectiveFactory.objective)
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
