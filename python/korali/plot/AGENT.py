#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt


# Plot AGENT results (read from .json files)
def plot(genList, **kwargs):

  lastGen = 0
  for i in genList:
    if genList[i]['Current Generation'] > lastGen:
      lastGen = genList[i]['Current Generation']


  data = genList[lastGen]
  numAgents = data["Problem"]["Agents Per Environment"]

  erStartSize = data["Solver"]["Experience Replay"]["Start Size"]
  expHistory = data["Solver"]["Training"]["Experience History"]
  offpHistory = data["Solver"]["Experience Replay"]["Off Policy"]["History"]
  bckW = data["Solver"]["Background Batch Importance Weight"]
  demW = data["Solver"]["Demonstration Batch Importance Weight"]
  cumExpHistory = np.cumsum(expHistory)
  
  ess  = data["Solver"]["Effective Sample Size"]
  ess = np.reshape(ess, ( -1,numAgents))

  demoFeatureRewards = data["Solver"]["Demonstration Feature Reward"]
  demoFeatureRewards = np.reshape(demoFeatureRewards, ( -1,numAgents))

  demoLogProbabilities = data["Solver"]["Demonstration Log Probability"]
  demoLogProbabilities = np.reshape(demoLogProbabilities, ( -1,numAgents))
  
  bckFeatureRewards = data["Solver"]["Background Feature Reward"]
  bckFeatureRewards = np.reshape(bckFeatureRewards, ( -1,numAgents))
  
  bckLogProbabilities = data["Solver"]["Background Log Probability"]
  bckLogProbabilities = np.reshape(bckLogProbabilities, ( -1,numAgents))
  
  maxEntropyObjective = data["Solver"]["Max Entropy Objective"]
  maxEntropyObjective = np.reshape(maxEntropyObjective, (-1,numAgents))
  
  logPartitionFunction = data["Solver"]["Log Partition Function History"]
  logPartitionFunction = np.reshape(logPartitionFunction, (-1,numAgents))

  fig, ax = plt.subplots(2, 2, num='Korali Results', figsize=(8, 8))
  ex = np.linspace(erStartSize, cumExpHistory[-1], len(demoFeatureRewards), endpoint=True)
  
  #ax[0,0].plot(cumExpHistory, offpHistory)
  #ax[0,0].plot(ex, bckZ, c='r')
  #ax[0,0].plot(ex, demZ, c='b')
  ax[0,0].plot(ex, ess, linewidth=1)
  ax[0,1].plot(ex, maxEntropyObjective, linewidth=1)
  ax[1,0].plot(ex, demoFeatureRewards, c='b', linewidth=1)
  ax[1,0].plot(ex, bckFeatureRewards, c='r', linewidth=1)
  ax[1,1].plot(ex, demoLogProbabilities, c='b', linewidth=1)
  ax[1,1].plot(ex, bckLogProbabilities, c='r', linewidth=1)

  #plt.suptitle('AGENTDiagnostics', fontweight='bold', fontsize=12)
