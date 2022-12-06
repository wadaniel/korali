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

  erStartSize = data["Solver"]["Experience Replay"]["Start Size"]
  expHistory = data["Solver"]["Training"]["Experience History"]
  offpHistory = data["Solver"]["Experience Replay"]["Off Policy"]["History"]
  ess  = data["Solver"]["Effective Sample Size"]
  bckZ = data["Solver"]["Background Batch Importance Weight"]
  demZ = data["Solver"]["Demonstration Batch Importance Weight"]
  cumExpHistory = np.cumsum(expHistory)

  demoFeatureRewards = data["Solver"]["Demonstration Feature Reward"]
  demoLogProbabilities = data["Solver"]["Demonstration Log Probability"]
  maxEntropyObjective = data["Solver"]["Max Entropy Objective"]

  fig, ax = plt.subplots(2, 2, num='Korali Results', figsize=(8, 8))
  ex = np.linspace(erStartSize, cumExpHistory[-1], len(demoFeatureRewards), endpoint=True)
  
  #ax[0,0].plot(cumExpHistory, offpHistory)
  #ax[0,0].plot(ex, bckZ, c='r')
  #ax[0,0].plot(ex, demZ, c='b')
  ax[0,0].plot(ex, ess)
  ax[0,1].plot(ex, maxEntropyObjective)
  ax[1,0].plot(ex, demoFeatureRewards)
  ax[1,1].plot(ex, demoLogProbabilities)

  #plt.suptitle('AGENTDiagnostics', fontweight='bold', fontsize=12)
