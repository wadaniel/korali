#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine
from scipy.signal import savgol_filter
import math 

def plot(genList, args):

 ##################### Plotting Rewards / Episode
 
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
 confidenceLevel = 2.326
 
 # Getting average cumulative reward statistics
 meanHistory = [ rewardHistory[0] ]
 confIntervalHistory = [ 0.0 ]
 for i in range(1, len(rewardHistory)):
  startPos = i - averageDepth
  if (startPos < 0): startPos = 0
  endPos = i
  data = rewardHistory[startPos:endPos]
  mean = np.mean(data)
  stdDev = np.std(data)
  confInterval = confidenceLevel * stdDev / math.sqrt(len(data))
  confIntervalHistory.append(confInterval)
  meanHistory.append(mean)
  
 meanHistory = np.array(meanHistory)
 confIntervalHistory = np.array(confIntervalHistory)
  
 epList = range(0, episodeCount) 
 fig1 = plt.figure()
 ax1 = fig1.add_subplot(111)
 ax1.set_ylabel('Cumulative Reward')  
 ax1.set_xlabel('Episode')
 ax1.set_title(solverName)
 ax1.plot(epList, rewardHistory, 'x', markersize=1, label='Episode Reward')
 ax1.plot(epList, meanHistory, '-', label='10 Episode Average')
 ax1.fill_between(epList, (meanHistory-confIntervalHistory), (meanHistory+confIntervalHistory), color='b', alpha=.1, label='98% Confidence Interval')
 ax1.hlines(trainingRewardThreshold, 0, episodeCount, linestyle='dashed', label='Training Threshold', color='red')
 ax1.hlines(testingRewardThreshold, 0, episodeCount, linestyle='dashdot', label='Testing Threshold', color='blue')
 ax1.legend(loc='lower right', ncol=1, fontsize=8)
 ax1.yaxis.grid()
 ax1.set_xlim([0, episodeCount-1])
 ax1.set_ylim([minPlotReward - 0.1*abs(minPlotReward), maxPlotReward + 0.1*abs(maxPlotReward)])

 ##################### (GFPT) Plotting Diagonal Covariance Values
 
 if ('GFPT' in agent['Solver']['Type']):
  varNames = [ ]
  for var in agent['Variables']:
   if (var['Type'] == 'Action'):
    varNames.append(var['Name'])
    
  varCovariances = [ ]
  for i in range(len(varNames)):
   varCovariances.append([ ])
   for gen in genList:
    varCovariances[i].append(genList[gen]['Solver']['Statistics']['Average Covariance'][i])
   
  genIds = [ ]
  for gen in genList:
   genIds.append(genList[gen]['Current Generation'])
   
  fig2 = plt.figure()
  ax2 = fig2.add_subplot(111)
  ax2.set_xticks(genIds)
  ax2.set_ylabel('Covariance Matrix Diagonal')  
  ax2.set_xlabel('Generation')
  ax2.set_title('GFPT - Cov Analysis')
  
  smoothWidth = 27;
  if (smoothWidth > len(genList)): smoothWidth = len(genList) - 1
  if ((smoothWidth % 2) == 0): smoothWidth = smoothWidth - 1 
  
  for i in range(len(varNames)):
   ax2.plot(savgol_filter(varCovariances[i], smoothWidth, 3), '-', label=varNames[i])
  
  ax2.legend(loc='lower right', ncol=1, fontsize=8)
  ax2.yaxis.grid()
  ax2.set_xlim([0, len(genList)-1])
    