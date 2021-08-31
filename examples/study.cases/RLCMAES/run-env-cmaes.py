#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

import argparse

# Parse console arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run', help='Run tag for result files.', required=True, type=int)
parser.add_argument('--dim', help='Dimension of objective function.', required=True, type=int)
parser.add_argument('--pop', help='CMAES population size.', required=True, type=int)
parser.add_argument('--obj', help='Objective function.', required=True, type=str)
parser.add_argument('--eval', help='Evaluate stored policy.', required=False, action='store_true')
parser.add_argument('--reps', help='Number of optimization runs.', required=False, default=1, type=int)

parser.add_argument('--noise', help='Noise level of objective function.', required=False, type=float, default=0.0)
args = parser.parse_args()
print(args)


# Importing computational model
import sys
import math
sys.path.append('./_environment')

from env import *

objective = args.obj
dim = args.dim
populationSize = args.pop
noise = args.noise
evaluation = args.eval
reps=args.reps

resultdir = "_env_cmaes_{}_{}_{}_{}".format(objective, dim, populationSize, noise, args.run)
 
maxSteps = 100

# Calculate defaults
mu = int(populationSize/2)
weights = np.log(mu+1/2)-np.log(np.array(range(mu))+1)
ueff = sum(weights)**2/sum(weights**2)
cs = (ueff+2.)/(dim+ueff+5)
cm = 1
action = [cs, cm]

outfile = "history_cmaes_{}_{}_{}_{}.npz".format(objective, dim, populationSize, noise)
objective = ObjectiveFactory(objective, dim, populationSize)

rewardhistory = np.zeros(reps)
for i in range(reps):
    objective.reset(noise=noise)
    state = objective.getState().tolist()
    step = 0
    cumreward = 0

    objectives = []
    muobjectives = []
    scales = []
    actions = []

    while step < maxSteps:

      # Performing the action
      done = objective.advance(action)
      
      # Getting Reward
      reward = objective.getReward()
      cumreward += reward
       
      # Storing New State
      state = objective.getState().tolist()
      
      # Advancing step counter
      objectives.append(objective.curBestF)
      muobjectives.append(objective.curEf)
      scales.append(objective.scale)
      actions.append(action)

      step = step + 1

     # Store statistics

    print(cumreward)
    rewardhistory[i] = cumreward
    if evaluation == True:
        if os.path.isfile(outfile):
            history = np.load(outfile)
            scaleHistory = history['scaleHistory']
            objectiveHistory = history['objectiveHistory']
            muobjectiveHistory = history['muobjectiveHistory']
            #actionHistory = history['actionHistory']

            scaleHistory = np.concatenate((scaleHistory, [scales]))
            objectiveHistory = np.concatenate((objectiveHistory, [objectives]))
            muobjectiveHistory = np.concatenate((objectiveHistory, [muobjectives]))
            #actionHistory = np.concatenate((objectiveHistory, [actions]))
        else:
            scaleHistory = [scales]
            objectiveHistory = [objectives]
            muobjectiveHistory = [muobjectives]
            #actionHistory = [actions]
         
        #np.savez(outfile, scaleHistory=scaleHistory, objectiveHistory=objectiveHistory, muobjectiveHistory=muobjectiveHistory, actionHistory=actionHistory)
        np.savez(outfile, scaleHistory=scaleHistory, objectiveHistory=objectiveHistory, muobjectiveHistory=muobjectiveHistory)

print("Mean & Sdev")
print(np.mean(rewardhistory))
print(np.std(rewardhistory))
