#!/usr/bin/env python3
import sys
sys.path.append('./_environment')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()


evaluation = False
resultDirectory = "_vracer_rosenbrock_new"
maxGens = 1e6
populationSize = 16
mu = int(populationSize/2) # states

### Defining the problem's configuration

e["Problem"]["Custom Settings"]["Evaluation"] = "False"

if evaluation == True:
    found = e.loadState(resultDirectory +'/latest')
    e["Problem"]["Custom Settings"]["Evaluation"] = "True"
    maxGens = maxGens + 1
    if found == False:
        sys.exit("Cannot run evaluation, results not found")

lEnv = lambda s : env(s,populationSize)

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lEnv
e["Problem"]["Testing Frequency"] = 100
e["Problem"]["Training Reward Threshold"] = np.inf
e["Problem"]["Policy Testing Episodes"] = 10
e["Problem"]["Actions Between Policy Updates"] = 10

for i in range(mu):
    e["Variables"][i*3]["Name"] = "Position X"
    e["Variables"][i*3]["Type"] = "State"

    e["Variables"][i*3+1]["Name"] = "Position Y"
    e["Variables"][i*3+1]["Type"] = "State"

    e["Variables"][i*3+2]["Name"] = "Evaluation"
    e["Variables"][i*3+2]["Type"] = "State"

i = 3*mu
e["Variables"][i]["Name"] = "Diagonal Variance 1"
e["Variables"][i]["Type"] = "State"

i += 1
e["Variables"][i]["Name"] = "Diagonal Variance 2"
e["Variables"][i]["Type"] = "State"

i += 1
e["Variables"][i]["Name"] = "Best Ever Evaluation"
e["Variables"][i]["Type"] = "State"

i += 1
e["Variables"][i]["Name"] = "Step Size Rate"
e["Variables"][i]["Type"] = "Action"
e["Variables"][i]["Lower Bound"] = 0.0
e["Variables"][i]["Upper Bound"] = +1.0
e["Variables"][i]["Initial Exploration Noise"] = 0.1

#i += 1
#e["Variables"][i]["Name"] = "Mean Learning Rate"
#e["Variables"][i]["Type"] = "Action"
#e["Variables"][i]["Lower Bound"] = 0.0
#e["Variables"][i]["Upper Bound"] = 1.0
#e["Variables"][i]["Initial Exploration Noise"] = 0.1

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 1

e["Solver"]["Experience Replay"]["Start Size"] = 1024
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch"]["Size"] = 128

e["Solver"]["State Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = maxGens
e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6


if evaluation == True:
    e["Solver"]["Testing"]["Sample Ids"] = [0]

### If this is test mode, run only a couple generations
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  e["Solver"]["Termination Criteria"]["Max Generations"] = 5

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 500
e["File Output"]["Path"] = resultDirectory

### Running Experiment

k.run(e)
