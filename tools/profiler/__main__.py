#!/usr/bin/env python3

import os
import sys
import json
import signal
import os.path
from os import path
import matplotlib
from matplotlib import cm
import argparse

signal.signal(signal.SIGINT, lambda x, y: exit(0))

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
labels = []
elapsedTime = js["Elapsed Time"]

for x in js["Timelines"]:
 if (not "Engine" in x):
  timeLists.append(js["Timelines"][x])
  labels.append(x)

# Calculating segment durations
startLists = []
durationLists = []
for i in range(len(timeLists)):
 startLists.append([])
 durationLists.append([])
 for j in range(0, len(timeLists[i]), 2):
  startLists[i].append(timeLists[i][j])
  durationLists[i].append(timeLists[i][j+1] - timeLists[i][j])
  
# Declaring a figure "gnt" 
pyplot.figure(1, figsize=(25, 10))
gnt = pyplot.subplot(211) 
  
# Setting Y-axis limits 
gnt.set_ylim(0, 10 + len(timeLists) * 10) 
  
# Setting X-axis limits 
gnt.set_xlim(0, elapsedTime) 
  
# Setting ticks on y-axis 
yticks = []
for i in range(len(timeLists)):
 yticks.append(10 + i*10)
 
gnt.set_yticks(yticks) 
gnt.set_yticklabels(labels) 
gnt.set_title('Worker Timelines')

# Setting graph attribute 
gnt.grid(False) 

for i in range(len(startLists)):
 segList = [ (startLists[i][j], durationLists[i][j]) for j in range(0, len(startLists[i])) ]
 gnt.broken_barh(segList, (yticks[i] - 5, 9), facecolors = ('tab:blue', 'blue') )
 
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

# Declaring a figure "gnt" 
gntEff = pyplot.subplot(212)
gntEff.set_ylim(0, 1.0)
gntEff.set_xlim(0, elapsedTime)

for y in ydims:  
 gntEff.plot(xdim, y)

gntEff.legend(labels)

# Setting labels for x-axis and y-axis
gntEff.set_title('Worker Efficiency')  
gntEff.set_xlabel('Time Elapsed (seconds)') 

pyplot.show() 