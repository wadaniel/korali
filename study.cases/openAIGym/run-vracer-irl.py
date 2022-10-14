#!/usr/bin/env python3
import sys
sys.path.append('./_model')

import json
import argparse

import numpy as np
from agent import *

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--run', help='Run number, used for output.', type=int, required=False, default=0)

args = parser.parse_args()
print(args)

####### Load observations

obsfile = f"observations_{args.env}.json"
obsstates = []
obsactions = []
with open(obsfile, 'r') as infile:
    obsjson = json.load(infile)
    obsstates = obsjson["States"]
    obsactions = obsjson["Actions"]

### Compute Feauters from states

#nf = 8
#nf = 17
obsfeatures = []
#maxFeatures = [-np.inf] * nf
for trajectory, actions in zip(obsstates, obsactions):
    features = []
    #distance = np.zeros(nf)
    for idx in range(len(trajectory)):
        # state: last two are velocities
        #f = list(trajectory[idx][-nf:]) #+ [float(sum(np.array(actions[idx])**2))]
        #features.append(list(f))
        features.append(list(trajectory[idx]))
        #distance += np.array(f)/0.04

    #print(f'distance covered: {distance}')
    obsfeatures.append(list(features))

print("Total observed trajectories: {}/{}".format(len(obsstates), len(obsactions)))
print("Max feature values found in observations:")
print(np.max(np.array(obsfeatures), axis=1))
print("Min feature values found in observations:")
print(np.min(np.array(obsfeatures), axis=1))
####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = f'_result_irl_{args.env}_{args.run}'
e.loadState(resultFolder + '/latest');

### Initializing openAI Gym environment

initEnvironment(e, args.env)

### IRL variables

e["Problem"]["Observations"]["States"] = obsstates
e["Problem"]["Observations"]["Actions"] = obsactions
e["Problem"]["Observations"]["Features"] = obsfeatures
e["Problem"]["Custom Settings"]["Print Step Information"] = "Enabled"

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Feature Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False

### IRL related configuration

e["Solver"]["Experiences Between Reward Updates"] = 500
e["Solver"]["Demonstration Batch Size"] = 5
e["Solver"]["Background Batch Size"] = 50
e["Solver"]["Background Sample Size"] = 500
e["Solver"]["Use Fusion Distribution"] = True
e["Solver"]["Experiences Between Partition Function Statistics"] = 1e5

## Reward Function Specification

e["Solver"]["Reward Function"]["Batch Size"] = 256
e["Solver"]["Reward Function"]["Learning Rate"] = 1e-4

e["Solver"]["Reward Function"]["L2 Regularization"]["Enabled"] = True
e["Solver"]["Reward Function"]["L2 Regularization"]["Importance"] = 1.

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 8

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 8

e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Reward Function"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"


### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6

### Setting file output configuration

e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 200
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)
