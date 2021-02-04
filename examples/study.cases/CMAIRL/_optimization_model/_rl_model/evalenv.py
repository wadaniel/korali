#!/usr/bin/env python3
import csv
import sys
import numpy as np
from cartpole import *

######## Defining Environment Storage

def evalenv(s):
 obsfile = s["Custom Settings"]["Input"]
 comparison = s["Custom Settings"]["Comparison"]
 states = []
 observation = []

 
 obsidx = -1
 if comparison == "action":
     obsidx = 4
 elif comparison == "position":
     obsidx = 7 # pole angle
 else:
     print("Error, 'Comparison' not recognized")
     sys.exit()
     #obsidx = 3 # pole velocity
 
 with open(obsfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    
    for row in csv_reader:
        states.append(row[:4])
        observation.append(row[obsidx])

 suml2 = 0.0

 for i, state in enumerate(states):

    s["State"] = state

    # Getting new action
    s.update()
 
    action = s["Action"]

    # Compare with observations
    reward = np.linalg.norm(np.array(observation[i])-np.array(action))
    s["Reward"] = reward

 # Done
 s["Termination"] = "Terminal"
