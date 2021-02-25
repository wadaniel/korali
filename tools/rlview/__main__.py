#! /usr/bin/env python3
import os
import sys
import signal
import json
import argparse
import time
import matplotlib
import importlib
import math 
import numpy as np
import matplotlib.pyplot as plt

from korali.plotter.helpers import hlsColors, drawMulticoloredLine
from scipy.signal import savgol_filter

##################### Plotting Reward History

def plotRewardHistory(ax, dirs, results, minReward, maxReward, averageDepth, maxObservations):

 confidenceLevel = 2.326 # 98%

 ## Setting initial x-axis (episode) and  y-axis (reward) limits
 
 maxPlotObservations = -math.inf
 maxPlotReward = -math.inf
 minPlotReward = +math.inf

 ## Creating colormap
 cmap = matplotlib.cm.get_cmap('coolwarm')
 colCurrIndex = 0.0

 ## Plotting the individual experiment results
    
 for resId, r in enumerate(results):
  
  if (len(r) == 0): continue  
  
  rewardHistory = r[-1]["Solver"]["Training"]["Reward History"]
  obsCountHistory = r[-1]["Solver"]["Training"]["Experience History"]
  
  # Gathering observation count for each result
  
  currObsCount = 0
  cumulativeObsList = []
  epList = range(0, len(rewardHistory)) 
  for c in obsCountHistory:
   currObsCount = currObsCount + c
   cumulativeObsList.append(currObsCount)
  
  # Updating common plot limits
 
  if (currObsCount > maxPlotObservations): maxPlotObservations = currObsCount
  if (maxObservations): maxPlotObservations = int(maxObservations)

  if (max(rewardHistory) > maxPlotReward): maxPlotReward = max(rewardHistory)
 
  trainingRewardThreshold = r[-1]["Problem"]["Training Reward Threshold"]
  testingRewardThreshold = r[-1]["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"]
 
  if (trainingRewardThreshold != math.inf): 
   if (trainingRewardThreshold > maxPlotReward): maxPlotReward = trainingRewardThreshold

  if (testingRewardThreshold != math.inf): 
   if (testingRewardThreshold > maxPlotReward): maxPlotReward = testingRewardThreshold
     
  if (min(rewardHistory) < minPlotReward): minPlotReward = min(rewardHistory)
  if (trainingRewardThreshold < minPlotReward): minPlotReward = trainingRewardThreshold
  if (testingRewardThreshold < minPlotReward): minPlotReward = testingRewardThreshold
 
  # Getting average cumulative reward statistics
  
  meanHistory = [ rewardHistory[0] ]
  confIntervalHistory = [ 0.0 ]
  for i in range(1, len(rewardHistory)):
   startPos = i - int(averageDepth)
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

  # Plotting common plot
  ax.plot(cumulativeObsList, rewardHistory, 'x', markersize=1.3, color=cmap(colCurrIndex), alpha=0.2, zorder=0)
  ax.plot(cumulativeObsList, meanHistory, '-', label=str(averageDepth) + '-Episode Average (' + dirs[resId] + ')', color=cmap(colCurrIndex), zorder=1)
  
  # Updating color index
  if (len(results) > 1): colCurrIndex = colCurrIndex + (1.0 / float(len(results)-1)) - 0.0001
  
 ## Configuring common plotting features
 
 if (minReward): minPlotReward = float(minReward)
 if (maxReward): maxPlotReward = float(maxReward)
 
 ax.set_ylabel('Cumulative Reward')  
 ax.set_xlabel('# Observations')
 ax.set_title('Korali RL History Viewer')
 
 ax.legend(loc='upper left', ncol=1, fontsize=8)
 ax.yaxis.grid()
 ax.set_xlim([0, maxPlotObservations-1])
 ax.set_ylim([minPlotReward - 0.1*abs(minPlotReward), maxPlotReward + 0.1*abs(maxPlotReward)])
 
 if (trainingRewardThreshold != math.inf): 
  ax.hlines(trainingRewardThreshold, 0, maxPlotObservations, linestyle='dashed', label='Training Threshold', color='red')

 if (testingRewardThreshold != math.inf): 
  ax.hlines(testingRewardThreshold, 0, maxPlotObservations, linestyle='dashdot', label='Testing Threshold', color='blue')

##################### Results parser

def parseResults(dir):

 results = [ ]
 for p in dir:
  configFile = p + '/latest'
  if (not os.path.isfile(configFile)):
    print(
        "[Korali] Error: Did not find any results in the {0} folder...".format(p))
    exit(-1)
 
  with open(configFile) as f:
    js = json.load(f)
  configRunId = js['Run ID']
 
  resultFiles = [
      f for f in os.listdir(p)
      if os.path.isfile(os.path.join(p, f)) and f.startswith('gen')
  ]
  resultFiles = sorted(resultFiles)
 
  genList = [ ]
 
  for file in resultFiles:
   if (not 'aux' in file):
    with open(p + '/' + file) as f:
      genJs = json.load(f)
      solverRunId = genJs['Run ID']
 
      if (configRunId == solverRunId):
        curGen = genJs['Current Generation']
        genList.append(genJs)
 
  del genList[0]
  results.append(genList)

 return results

##################### Main Routine: Parsing arguments and result files
  
if __name__ == '__main__':
 
 # Setting termination signal handler
 
 signal.signal(signal.SIGINT, lambda x, y: exit(0))

 # Parsing arguments

 parser = argparse.ArgumentParser(
     prog='korali.rlview',
     description='Plot the results of a Korali Reinforcement Learning execution.')
 parser.add_argument(
     '--dir',
     help='Path(s) to result files, separated by space',
     default=['_korali_result'],
     required=False,
     nargs='+')
 parser.add_argument(
     '--maxObservations',
     help='Maximum observations (x-axis) to display',
     default=None,
     required=False)
 parser.add_argument(
     '--maxReward',
     help='Maximum reward to display',
     default=None,
     required=False)
 parser.add_argument(
     '--updateFrequency',
     help='Specified the time (seconds) between live updates to the plot',
     default=0.0,
     required=False)
 parser.add_argument(
     '--minReward',
     help='Minimum reward to display',
     default=None,
     required=False)
 parser.add_argument(
      '--check',
      help='Verifies that the module has been installed correctly',
      action='store_true',
      required=False)
 parser.add_argument(
      '--averageDepth',
      help='Specifies the depth for plotting average',
      default=1000,
      required=False)
 parser.add_argument(
      '--test',
      help='Run without graphics (for testing purpose)',
      action='store_true',
      required=False)
 args = parser.parse_args()

 ### Checking installation
 
 if (args.check == True):
  print("[Korali] RL Viewer correctly installed.")
  exit(0)
 
 ### Setup without graphics, if needed
 
 if (args.test): matplotlib.use('Agg')
 
 ### Reading values from result files

 results = parseResults(args.dir)
  
 ### Creating figure(s)
  
 fig1 = plt.figure()
 ax1 = fig1.add_subplot(111)
     
 ### Creating plots
     
 plotRewardHistory(ax1, args.dir, results, args.minReward, args.maxReward, args.averageDepth, args.maxObservations)
 plt.draw()
 
 ### Printing live results if update frequency > 0
 
 fq = float(args.updateFrequency)
 if (fq > 0.0):
  while(True):
   results = parseResults(args.dir)
   plt.pause(fq)
   ax1.clear()
   plotRewardHistory(ax1, args.dir, results, args.minReward, args.maxReward, args.averageDepth, args.maxObservations)
   plt.draw()
   
 plt.show() 