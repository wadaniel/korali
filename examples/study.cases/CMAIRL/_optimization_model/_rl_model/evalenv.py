#!/usr/bin/env python3
import csv
from cartpole import *

######## Defining Environment Storage

def evalenv(s, th):

 obsfile = 'observations.csv'
 states = []
 obsactions = []

 with open(obsfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        states.append(row[:4])
        obsactions.append(row[4])

 suml2 = 0.0

 for i, state in enumerate(states):

    s["State"] = state

    # Getting new action
    s.update()
 
    action = s["Action"]

    # Compare with observations
    s["Reward"] = np.sum(np.power((np.array(obsactions[i])-np.array(action)),2))
