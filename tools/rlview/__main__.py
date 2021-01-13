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

def plotRewardHistory(ax, results, minReward, maxReward, averageDepth):

 lastGenId = 0
  
 ## Finding last generation among all experiments
 for r in results: 
    if r[-1]['Current Generation'] > lastGenId:
     lastGenId = r[-1]['Current Generation']

 ## Finding x-axis (reward) limits
 
 maxPlotEpisode = -math.inf
 minPlotEpisode = 0
 
 for r in results:
  episodeCount = len(r[-1]["Solver"]["Training"]["Reward History"])
  if (episodeCount > maxPlotEpisode): maxPlotEpisode = episodeCount
  
 ## Finding y-axis (reward) limits

 maxPlotReward = -math.inf
 minPlotReward = +math.inf
   
 for r in results:
 
  rewardHistory = results[0][-1]["Solver"]["Training"]["Reward History"]
 
  trainingRewardThreshold = r[-1]["Problem"]["Training Reward Threshold"]
  testingRewardThreshold = r[-1]["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"]
  
  if (max(rewardHistory) > maxPlotReward): maxPlotReward = max(rewardHistory)
  if (trainingRewardThreshold > maxPlotReward): maxPlotReward = trainingRewardThreshold
  if (testingRewardThreshold > maxPlotReward): maxPlotReward = testingRewardThreshold
  
  if (min(rewardHistory) < minPlotReward): minPlotReward = min(rewardHistory)
  if (trainingRewardThreshold < minPlotReward): minPlotReward = trainingRewardThreshold
  if (testingRewardThreshold < minPlotReward): minPlotReward = testingRewardThreshold
 
 if (minReward): minPlotReward = float(minReward)
 if (maxReward): maxPlotReward = float(maxReward)
 
 # Setting average depth
 
 confidenceLevel = 2.326
 
 ## Plotting the individual experiment results

 for r in results:
  
  # Gathering result information
  
  solverName = r[-1]["Solver"]["Type"]
  rewardHistory = r[-1]["Solver"]["Training"]["Reward History"]
 
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

  # Plotting result
    
  epList = range(0, len(rewardHistory)) 
  ax.plot(epList, rewardHistory, 'x', markersize=1, label='Episode Reward (' + solverName + ')')
  ax.plot(epList, meanHistory, '-', label=str(averageDepth) + '-Episode Average (' + solverName + ')')
  ax.fill_between(epList, (meanHistory-confIntervalHistory), (meanHistory+confIntervalHistory), color='b', alpha=.1, label='98% Confidence Interval (' + solverName + ')')
  
 ## Configuring common plotting features
 
 ax.set_ylabel('Cumulative Reward')  
 ax.set_xlabel('Episode')
 ax.set_title('Korali RL History Viewer')
 ax.hlines(trainingRewardThreshold, 0, episodeCount, linestyle='dashed', label='Training Threshold', color='red')
 ax.hlines(testingRewardThreshold, 0, episodeCount, linestyle='dashdot', label='Testing Threshold', color='blue')
 ax.legend(loc='lower right', ncol=1, fontsize=8)
 ax.yaxis.grid()
 ax.set_xlim([minPlotEpisode, maxPlotEpisode-1])
 ax.set_ylim([minPlotReward - 0.1*abs(minPlotReward), maxPlotReward + 0.1*abs(maxPlotReward)])

##################### (GFPT) Plotting Diagonal Covariance Values

def plotGFPTCovariance(ax, results):
 varNames = [ ]
 for var in results[0][0]['Variables']:
  if (var['Type'] == 'Action'):
   varNames.append(var['Name'])
   
 varCovariances = [ ]
 for i in range(len(varNames)):
  varCovariances.append([ ])
  for gen in results[0]:
   varCovariances[i].append(gen['Solver']['Statistics']['Average Covariance'][i])
  
 genIds = [ ]
 for gen in results[0]:
  genIds.append(gen['Current Generation'])
  
 ax.set_xticks(genIds)
 ax.set_ylabel('Covariance Matrix Diagonal')  
 ax.set_xlabel('Generation')
 ax.set_title('GFPT - Cov Analysis')
 
 smoothWidth = 27;
 if (smoothWidth > len(results[0])): smoothWidth = len(results[0]) - 1
 if ((smoothWidth % 2) == 0): smoothWidth = smoothWidth - 1 
 
 for i in range(len(varNames)):
  ax.plot(savgol_filter(varCovariances[i], smoothWidth, 3), '-', label=varNames[i])
 
 ax.legend(loc='lower right', ncol=1, fontsize=8)
 ax.yaxis.grid()
 ax.set_xlim([0, len(results[0])-1])

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
      default=10,
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
 solverName = results[0][0]['Solver']['Type']
  
 ### Creating figure(s)
  
 fig1 = plt.figure()
 ax1 = fig1.add_subplot(111)
 
 if ('GFPT' in results[0][0]['Solver']['Type']):
  fig2 = plt.figure()
  ax2 = fig2.add_subplot(111) 
     
 ### Creating plots
     
 plotRewardHistory(ax1, results, args.minReward, args.maxReward, int(args.averageDepth))
 if ('GFPT' in solverName): plotGFPTCovariance(ax2, results) 
 plt.draw()
 
 ### Printing live results if update frequency > 0
 
 fq = float(args.updateFrequency)
 if (fq > 0.0):
  while(True):
   results = parseResults(args.dir)
   plt.pause(fq)
   ax1.clear()
   plotRewardHistory(ax1, results, args.minReward, args.maxReward, averageDepth)
   if ('GFPT' in solverName):
    ax2.clear()
    plotGFPTCovariance(ax2, results)
   plt.draw()
   
 plt.show() 
