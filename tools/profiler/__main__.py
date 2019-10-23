#!/usr/bin/env python3

import os
import sys
import json
import os.path
from os import path
import matplotlib
from matplotlib import cm
import argparse


parser = argparse.ArgumentParser(prog='korali.plotter', description='Show profiling information of a Korali execution.')
parser.add_argument('--dir', help='Directory of result files', default='_korali_result', required = False)
parser.add_argument('--test', help='Run without graphics (for testing purpose)', action='store_true', required = False)
args = parser.parse_args()

filePath = args.dir + '/profiling.json'
if (not path.exists(filePath) ):
 print('[Korali] Error: Could not find result file "profiling.json" in folder: ' + args.dir + '.')
 exit(-1)
 
if (args.test == True):
 print('[Korali] Testing profiler plotter for file: ' + filePath + '...') 
 matplotlib.use('Agg')

from matplotlib import pyplot

with open(filePath) as f:
 jsString = f.read()
js  = json.loads(jsString)

timeLists = []
idLists = []
labels = []
elapsedTime = js["Elapsed Time"]

for x in js["Timelines"]:
 timeLists.append(js["Timelines"][x]["Timestamp"])
 idLists.append(js["Timelines"][x]["Segment Id"])
 labels.append(x)

maxSampleId = -1
maxSolverId = -1 
for i in range(len(timeLists)):
 if ("Worker" in labels[i]):
  myMaxSampleId = max(idLists[i]) 
  if (myMaxSampleId > maxSampleId): maxSampleId = myMaxSampleId;
 
 if ("Engine" in labels[i]):
  myMaxSolverId = max(idLists[i]) 
  if (myMaxSolverId > maxSolverId): maxSolverId = myMaxSolverId;

sampleColors = cm.get_cmap('viridis', maxSampleId + 1)
solverColors = cm.get_cmap('viridis', maxSolverId + 1)

# Declaring a figure "gnt" 
fig, gnt = pyplot.subplots() 
  
# Setting Y-axis limits 
gnt.set_ylim(0, 10 + len(timeLists) * 10) 
  
# Setting X-axis limits 
gnt.set_xlim(0, elapsedTime) 
  
# Setting labels for x-axis and y-axis 
gnt.set_xlabel('Time Elapsed (seconds)') 

# Setting ticks on y-axis 
yticks = []
for i in range(len(timeLists)):
 yticks.append(10 + i*10)
 
gnt.set_yticks(yticks) 
gnt.set_yticklabels(labels) 

# Setting graph attribute 
gnt.grid(False) 

for i in range(len(timeLists)):
 segList = []
 segColor = []
 #print(len(list))
 for j in range(0, len(timeLists[i]), 2):
  segList.append((timeLists[i][j], timeLists[i][j+1] - timeLists[i][j])) 
  #if ("Worker" in labels[i]): segColor.append(sampleColors.colors[idLists[i]])
  #if ("Engine" in labels[i]): segColor.append(solverColors.colors[idLists[i]])
 gnt.broken_barh(segList, (yticks[i] - 5, 9), facecolors = ('tab:blue', 'blue') )
 
pyplot.show() 