#!/usr/bin/env python3
import os
from objective import *

######## Defining Environment Storage


trainingObjectiveList = ["fsphere", "felli", "fcigar", "ftablet", "fcigtab", "ftwoax", "fdiffpow", "rosenbrock", "fparabr", "fsharpr"]
evaluationObjectiveList = ["fsphere", "felli", "fcigar", "ftablet", "fcigtab", "ftwoax", "fdiffpow", "rosenbrock", "fparabr", "fsharpr", "booth", "dixon", "ackley", "levi", "rastrigin"]
 
def env(s, objective, dim, populationSize, maxSteps, noise, version):

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

 objectives = np.zeros(maxSteps)
 meanobjectives = np.zeros(maxSteps)
 scales = np.zeros(maxSteps)
 actions = np.zeros((maxSteps, 3))
 paths = np.zeros(maxSteps)

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = objectiveFactory.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = objectiveFactory.getReward()
   
  # Storing New State
  s["State"] = objectiveFactory.getState().tolist()
  
  # Store statistics
  if s["Custom Settings"]["Evaluation"] == "True":
    objectives[step] = objectiveFactory.curBestF
    meanobjectives[step] = objectiveFactory.curEf
    scales[step] = objectiveFactory.scale
    paths[step] = objectiveFactory.pathsNorm
    actions[step, :] = s["Action"]

  # Advancing step counter
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
        pathsHistory = history['pathsHistory']
        objectiveHistory = history['objectiveHistory']
        meanobjectiveHistory = history['meanobjectiveHistory']
        actionHistory = history['actionHistory']

        scaleHistory = np.concatenate((scaleHistory, [scales]))
        pathsHistory = np.concatenate((pathsHistory, [paths]))
        objectiveHistory = np.concatenate((objectiveHistory, [objectives]))
        meanobjectiveHistory = np.concatenate((objectiveHistory, [meanobjectives]))
        actionHistory = np.concatenate((actionHistory, [actions]))

    else:
        scaleHistory = [scales]
        pathsHistory = [paths]
        objectiveHistory = [objectives]
        meanobjectiveHistory = [meanobjectives]
        actionHistory = [actions]
     
    np.savez(outfile, scaleHistory=scaleHistory, pathsHistory=pathsHistory, objectiveHistory=objectiveHistory, meanobjectiveHistory=meanobjectiveHistory, actionHistory=actionHistory)
