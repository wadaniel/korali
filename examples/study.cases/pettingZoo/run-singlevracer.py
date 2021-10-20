#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent_single import *
import pdb

####### Parsing arguments

os.environ["SDL_VIDEODRIVER"] = "dummy"
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument('--dis', help='Sampling Distribution.', required=False,type = str, default = 'Clipped Normal')
parser.add_argument('--l2', help='L2 Regularization.', required=False, type=float, default = 0.)
parser.add_argument('--opt', help='Off Policy Target.', required=False, type=float, default = 0.1)
parser.add_argument('--lr', help='Learning Rate.', required=False, type=float, default = 0.0001)
parser.add_argument('--run', help='Run Number', required=True, type=int, default = 0)
parser.add_argument('--model', help='Model Number', required=False, type=str, default = '')
#model  '' 1 agent 1 action , model '1' 5 agents with [oi,o0,o1,o2,o3,o4] as input, model '2' [o_i, random]
parser.add_argument('--num_layers', help='Hidden Layer Number', required=False, type=int, default = 128)
args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

dis_dir = args.dis.replace(" ","_")
resultFolder = 'results/_result_vracer_single' + args.env + '_' + dis_dir + '_' + str(args.num_layers)+ '_' + str(args.run) +'/'
e.loadState(resultFolder + '/latest');


### Initializing openAI Gym environment

initEnvironment(e, args.env, args.model)

###make text file to write used args


### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = args.lr
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = args.opt

e["Solver"]["Policy"]["Distribution"] = args.dis
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True


  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = args.l2 > 0.
e["Solver"]["L2 Regularization"]["Importance"] = args.l2

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.num_layers

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.num_layers

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = 10e6
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 100
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)

file = open(resultFolder + 'args.txt',"w")
file.write(str(args))
file.close()