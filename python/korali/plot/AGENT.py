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
  #offpHistory = data["Solver"]["Experience Replay"]["Off Policy"]["History"]
  cumExpHistory = np.cumsum(expHistory)

  demoFeatureRewards = data["Solver"]["Demonstration Feature Reward"]
  demoLogProbabilities = data["Solver"]["Demonstration Log Probability"]

  fig, ax = plt.subplots(2, 2, num='Korali Results', figsize=(8, 8))
  
  ax[0,1].plot(cumExpHistory, offpHistory)

  ex = np.linspace(erStartSize, cumExpHistory[-1], len(demoFeatureRewards), endpoint=True)
  ax[1,0].plot(ex, demoFeatureRewards)
  ex = np.linspace(erStartSize, cumExpHistory[-1], len(demoLogProbabilities), endpoint=True)
  ax[1,1].plot(ex, demoLogProbabilities)

  #plt.suptitle('AGENTDiagnostics', fontweight='bold', fontsize=12)
