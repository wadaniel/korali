#!/usr/bin/env python3

import os
import sys
import json
import signal
import os.path
from os import path
import matplotlib
from matplotlib import cm
import matplotlib.colors as colors
import argparse
import numpy as np

signal.signal(signal.SIGINT, lambda x, y: exit(0))

parser = argparse.ArgumentParser(prog='korali.plotter', description='Show profiling information of a Korali execution.')
parser.add_argument('--file', help='Json file with profiling information to read.', default='./profiling.json', required = False)
parser.add_argument('--test', help='Run without graphics (for testing purpose)', action='store_true', required = False)
parser.add_argument('--tend', help='Indicates the maximum time to report in the timeline', required = False, default = 'undefined')
parser.add_argument('--plot', help='Indicates the type of plot to generate', choices=['timeline', 'efficiency'], required = True)

args = parser.parse_args()

if (not path.exists(args.file) ):
 print('[Korali] Error: Could not find profiling information file: ' + args.file + '.')
 exit(-1)
 
if (args.test == True):
 print('[Korali] Testing profiler plotter for file: ' + args.file + '...') 
 matplotlib.use('Agg')

from matplotlib import pyplot

with open(args.file) as f:
 jsString = f.read()
js  = json.loads(jsString)

timelines = []
labels = []
experimentCount = js["Experiment Count"]
elapsedTime = js["Elapsed Time"]

if (args.tend == 'undefined'): tend = elapsedTime
else: tend = float(args.tend) 

for x in js["Timelines"]:
  timelines.append(js["Timelines"][x])
  labels.append(x)

fig, ax = pyplot.subplots(1, 1, sharex=True, figsize=(25, 10))

######################## Preprocessing information

# Calculating segment durations
startLists = []
durationLists = []
solverIdLists = []
for list in timelines:
 currentStartList = []
 currentDurationList = []
 currentSolverIdList = []
 for segment in list:
  currentStartList.append(segment["Start Time"])
  currentDurationList.append(segment["End Time"] - segment["Start Time"])
  currentSolverIdList.append(segment["Solver Id"])
 startLists.append(currentStartList)
 durationLists.append(currentDurationList)
 solverIdLists.append(currentSolverIdList)
  
######################## Creating Time-based figure

#### Creating Timeline Plot

if (args.plot == 'timeline'):


 # Setting Y-axis limits 
 upperLimit = 10 + len(timelines) * 10
 ax.set_ylim(0, upperLimit) 
  
 # Setting X-axis limits 
 ax.set_xlim(0, tend) 
  
 # Setting ticks on y-axis 
 yticks = [] 
 for i in range(len(timelines)):
  yticks.append(10 + i*10)
 xticks = np.arange(0, 21, 2)
 ax.set_xticks(xticks)

 xticklabels = []
 for tick in xticks: xticklabels.append("{:.1f}".format(tick))
 ax.set_xticklabels(xticklabels)

 # Setting graph attribute 
 ax.grid(False) 

 # Creating Color list
 cMap = pyplot.get_cmap('viridis')
 cNorm  = colors.Normalize(vmin=0, vmax=16)
 scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cMap)
 colorMap = [ scalarMap.to_rgba(i) for i in range(16) ]
 colorMap.reverse()

 for i in range(len(startLists)):
  #colorList = [ 'tab:blue', 'blue' ]
  #solverList = [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9, 12, 12, 12, 12, 12]
  #colorList = [ colorMap[solverList[j]] for j in range(0, len(solverIdLists[i])) ]
  colorList = [ colorMap[solverIdLists[i][j]*3] for j in range(0, len(solverIdLists[i])) ]
  segList = [ (startLists[i][j], durationLists[i][j]) for j in range(0, len(startLists[i])) ]
  ax.broken_barh(segList, (yticks[i] - 5, 9), facecolors = tuple(colorList), edgecolor = 'black' )

#### Creating Efficiency Plot

if (args.plot == 'efficiency'): 
 N = 1000
 dt =  elapsedTime / N

 xdim = []
 for i in range(1, N):
  xdim.append(dt*i)

 ydims = []
 for i in range(len(startLists)):
  ydims.append([])
  for ct in xdim:
   busyTime = 0.0
   for j in range(len(startLists[i])):
    if (startLists[i][j] > ct):
     break;
    if (startLists[i][j] + durationLists[i][j] > ct): 
     busyTime += ct - startLists[i][j];
     break;  
    if (startLists[i][j] + durationLists[i][j] < ct): 
     busyTime += durationLists[i][j];
   efficiency = busyTime / ct;
   ydims[i].append(efficiency)

 averageEfficiency = [] 
 for j in range(len(xdim)): 
  totalEfficiency = 0.0
  for i in range(len(startLists)):
   totalEfficiency += ydims[i][j]
  averageEfficiency.append(totalEfficiency / len(startLists))
 
 minLimit = 0.0
 maxLimit = 1.02
 if (maxLimit > 1.0): maxLimit = 1.02
 if (minLimit < 0.0): minLimit = 0.0
 ax.set_ylim(minLimit, maxLimit)
 ax.set_xlim(0, tend)
 effLabels = []
 if (len(ydims) < 10):
  for y in ydims: ax.plot(xdim, y)
  effLabels = labels

 yticks = np.arange(0, 1.01, 0.1)
 ax.set_yticks(yticks)

 ytickLabels = []
 for tick in yticks: ytickLabels.append("{:.0f}%".format(tick*100))
 ax.set_yticklabels(ytickLabels)

 ax.set_ylabel('Core Usage', fontsize=20)
 ax.set_yticks([])
 ax.plot(xdim, averageEfficiency, '-', linewidth=4)
 ax.set_ylabel('Average Efficiency', fontsize=16)  
 ax.set_xlabel('Time', fontsize=16)  

####################### Creating Plots

pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.show() 
