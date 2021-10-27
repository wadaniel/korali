#!/usr/bin/env python3
import sys
sys.path.append('./_environment')
from env import *

import argparse

# Parse console arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run', help='Run tag for result files.', required=True, type=int)
parser.add_argument('--obj', help='Objective function.', required=True, type=str)
parser.add_argument('--dim', help='Dimension of objective function.', required=True, type=int)
parser.add_argument('--pop', help='CMAES population size.', required=True, type=int)
parser.add_argument('--eval', help='Evaluate stored policy.', required=False, action='store_true')

# Defaults
parser.add_argument('--exp', help='VRACER max experiences.', required=False, type=int, default=1000000)
parser.add_argument('--noise', help='Noise level of objective function.', required=False, type=float, default=0.0)

args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

# Experiment Configuration

run = args.run
objective = args.obj
dim = args.dim
evaluation = args.eval
populationSize = args.pop
noise = args.noise
maxExperiences = args.exp
resultDirectory = "_vracer_rnn_{}_{}_{}_{}_{}".format(objective, dim, populationSize, noise, run)

if objective == "random":
    environmentCount = 7
else:
    environmentCount = 1

mu = int(populationSize/2) # states

# Termination Criteria


### Defining the problem's configuration

e["Problem"]["Custom Settings"]["Evaluation"] = "False"

if evaluation == True:
    found = e.loadState(resultDirectory +'/latest')
    e["Problem"]["Custom Settings"]["Evaluation"] = "True"
    maxGens = int(e["Current Generation"]) + 1
    if found == False:
        sys.exit("Cannot run evaluation, results not found")

lEnv = lambda s : env(s, objective, dim, populationSize, noise)

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lEnv
e["Problem"]["Environment Count"] = environmentCount
e["Problem"]["Testing Frequency"] = 500
e["Problem"]["Training Reward Threshold"] = np.inf
e["Problem"]["Policy Testing Episodes"] = 10
e["Problem"]["Actions Between Policy Updates"] = 0.2

i = 0
for j in range(mu):
    for d in range(dim):
        e["Variables"][i]["Name"] = "Position {}/{}".format(j,d)
        e["Variables"][i]["Type"] = "State"
        i += 1
    
    e["Variables"][i]["Name"] = "Evaluation"
    e["Variables"][i]["Type"] = "State"
    i += 1

e["Variables"][i]["Name"] = "Best Ever Evaluation"
e["Variables"][i]["Type"] = "State"
i += 1

e["Variables"][i]["Name"] = "Step Size Rate"
e["Variables"][i]["Type"] = "Action"
e["Variables"][i]["Lower Bound"] = 0.0
e["Variables"][i]["Upper Bound"] = +1.0
e["Variables"][i]["Initial Exploration Noise"] = 0.2
i += 1

e["Variables"][i]["Name"] = "Mean Adaption Rate"
e["Variables"][i]["Type"] = "Action"
e["Variables"][i]["Lower Bound"] = 0.0
e["Variables"][i]["Upper Bound"] = +1.0
e["Variables"][i]["Initial Exploration Noise"] = 0.2
i += 1

#e["Variables"][i]["Name"] = "Damping param"
#e["Variables"][i]["Type"] = "Action"
#e["Variables"][i]["Lower Bound"] = 1
#e["Variables"][i]["Upper Bound"] = 3
#e["Variables"][i]["Initial Exploration Noise"] = 0.2
#i += 1


### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 10

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch"]["Size"] = 128

e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/LSTM"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Depth"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Experiences"] = maxExperiences

if evaluation == True:
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = list(range(10))

### If this is test mode, run only a couple generations
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  e["Solver"]["Termination Criteria"]["Max Generations"] = 5

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1000
e["File Output"]["Path"] = resultDirectory
e["Console Output"]["Verbosity"] = "Detailed"

### Running Experiment

k.run(e)