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
import scipy.stats as st
import matplotlib.pyplot as plt

from korali.plotter.helpers import hlsColors, drawMulticoloredLine
from scipy.signal import savgol_filter

##################### Plotting Reward History

def plotIRL(axs, dirs, results ):

 ## Creating colormap
 cmap = matplotlib.cm.get_cmap('brg')
 colCurrIndex = 0.0

 ## Plotting the individual experiment results
  
 for idx, r in enumerate(results):

    print("Processing {} ..".format(dirs[idx]))
    experienceCount = []
    featureWeights = []
    featureWeightsGradient = []
    logPartitionFunction = []
    logPartitionFunctionSdev = []
 
    for s in r:
        experienceCount.append(s['Solver']['Experience Count'])
        featureWeights.append(s['Solver']['Feature Weights'])
        featureWeightsGradient.append(s['Solver']['Feature Weight Gradient'])
        logPartitionFunction.append(s['Solver']['Log Partition Function'])
        logPartitionFunctionSdev.append(s['Solver']['Log Sdev Partition Function'])
  
    dtext = str(dirs[idx])
    labeltxt = "dir" + dtext
    axs[0, 0].plot(experienceCount, logPartitionFunction)
    axs[1, 0].plot(experienceCount, logPartitionFunctionSdev)
    axs[0, 1].plot(experienceCount, featureWeights, label=labeltxt)
    axs[1, 1].plot(experienceCount, featureWeightsGradient)

 ## Configuring common plotting features
 axs[0, 0].set_title('Log Partition Function')
 axs[1, 0].set_title('Log Standard Deviation Partition Function')
 axs[0, 1].set_title('Feature Weights')
 axs[1, 1].set_title('Gradient Llk wrt. Feature Weights')
 axs[1, 0].set_xlabel('# Observations')
 axs[1, 1].set_xlabel('# Observations')
 #plt.legend(axs[0,1])
 axs[0, 1].legend()

##################### Results parser

def parseResults(dir):

 results = [ ]
 
 for p in dir:
  configFile = p + '/latest'
  if (not os.path.isfile(configFile)):
    print("[Korali] Error: Did not find any results in the {0} folder...".format(p))
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
      '--check',
      help='Verifies that the module has been installed correctly',
      action='store_true',
      required=False)
 parser.add_argument(
      '--test',
      help='Run without graphics (for testing purpose)',
      action='store_true',
      required=False)
 args = parser.parse_args()

 ### Checking installation
 
 if (args.check == True):
  print("[Korali] IRL Viewer correctly installed.")
  exit(0)
 
 ### Setup without graphics, if needed
 
 if (args.test): matplotlib.use('Agg')
 
 ### Reading values from result files

 results = parseResults(args.dir)
 if (len(results) == 0): 
  print('Error: No result folders have been provided for plotting.')
  exit(-1)
 
 ### Creating figure(s)
  
 fig, axs = plt.subplots(2, 2)
     
 ### Creating plots
   
 print("Number of directories parsed: {}".format(len(results)))
 print("Number of files per directory: {}".format([len(f) for f in results]))
 plotIRL(axs, args.dir, results)

 plt.show() 
