#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine
import math 

def plot(genList, args):

 lastGenId = 0
 for i in genList:
   if genList[i]['Current Generation'] > lastGenId:
     lastGenId = genList[i]['Current Generation']
 agent = genList[lastGenId]
 
 solverName = agent["Solver"]["Type"]
 trainingRewardThreshold = agent["Problem"]["Training Reward Threshold"]
 testingRewardThreshold = agent["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"]     
 rewardHistory = agent["Solver"]["Training"]["Reward History"]
 
 maxPlotReward = -math.inf
 if (max(rewardHistory) > maxPlotReward): maxPlotReward = max(rewardHistory)
 if (trainingRewardThreshold > maxPlotReward): maxPlotReward = trainingRewardThreshold
 if (testingRewardThreshold > maxPlotReward): maxPlotReward = testingRewardThreshold

 minPlotReward = +math.inf
 if (min(rewardHistory) < minPlotReward): minPlotReward = min(rewardHistory)
 if (trainingRewardThreshold < minPlotReward): minPlotReward = trainingRewardThreshold
 if (testingRewardThreshold < minPlotReward): minPlotReward = testingRewardThreshold
 
 # Getting episode Count
 episodeCount = len(rewardHistory)
 
 # Setting average depth
 averageDepth = 10
 
 # Getting average cumulative reward statistics
 averageHistory = [ ]
 expCount = 0
 rewardSum = 0
 for i, r in enumerate(rewardHistory):
  rewardSum = rewardSum + r
  expCount = expCount + 1
 
  # Adjusting for moving average 
  if (i >= averageDepth):
    expCount = averageDepth
    rewardSum = rewardSum - rewardHistory[i - averageDepth]
    
  curAverage = rewardSum / expCount;
  averageHistory.append(curAverage)
 
 fig = plt.figure()
 ax = fig.add_subplot(111)
 ax.set_ylabel('Cumulative Reward')  
 ax.set_xlabel('Episode')
 ax.set_title(solverName)
 ax.plot(rewardHistory, 'x', markersize=1, label='Episode Reward')
 ax.plot(averageHistory, '-', label='10-Episode Average')
 ax.hlines(trainingRewardThreshold, 0, episodeCount, linestyle='--', label='Training Threshold')
 ax.hlines(testingRewardThreshold, 0, episodeCount, linestyle='--', label='Testing Threshold')
 
 plt.xlim([0, episodeCount-1])
 plt.ylim([minPlotReward - 0.1*abs(minPlotReward), maxPlotReward + 0.1*abs(maxPlotReward)])
 ax.yaxis.grid()

 ax.legend(loc='lower right', ncol=1, fontsize=8)  
