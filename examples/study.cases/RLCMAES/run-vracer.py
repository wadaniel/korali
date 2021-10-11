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
parser.add_argument('--feval', help='Evaluate stored policy.', required=False, type=str)
parser.add_argument('--reps', help='Number of optimization runs.', required=False, default=10, type=int)

# Defaults
parser.add_argument('--exp', help='VRACER max experiences.', required=False, type=int, default=1000000)
parser.add_argument('--noise', help='Noise level of objective function.', required=False, type=float, default=0.0)
parser.add_argument('--version', help='Version of objective factory.', required=False, type=int, default=0)

args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

# Experiment Configuration

run = args.run
objective = args.obj
fobjective = args.feval if args.feval else objective
dim = args.dim
populationSize = args.pop
noise = args.noise
maxExperiences = args.exp
evaluation = args.eval
repetitions = args.reps
version = args.version

resultDirectory = "_vracer_{}_{}_{}_{}_{}".format(objective, dim, populationSize, noise, run)

if objective == "random":
    environmentCount = len(objectiveList)
else:
    environmentCount = 1

mu = int(populationSize/4) # states

### Defining the problem's configuration

e["Problem"]["Custom Settings"]["Evaluation"] = "False"

if evaluation == True:
    found = e.loadState(resultDirectory +'/latest')
    outfile = "history_vracer_{}_{}_{}_{}_{}.npz".format(objective, dim, populationSize, noise, run)
    e["Problem"]["Custom Settings"]["Evaluation"] = "True"
    e["Problem"]["Custom Settings"]["Outfile"] = outfile
    maxGens = int(e["Current Generation"]) + 1
    if found == False:
        sys.exit("Cannot run evaluation, results not found")

lEnv = lambda s : env(s, fobjective, dim, populationSize, noise, version)

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lEnv
e["Problem"]["Environment Count"] = environmentCount
e["Problem"]["Testing Frequency"] = 500
e["Problem"]["Training Reward Threshold"] = np.inf
e["Problem"]["Policy Testing Episodes"] = 50

if version == 0:
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
else:
    i = 0
    for j in range(dim):
        e["Variables"][i]["Name"] = "Mu Cov {}".format(j)
        e["Variables"][i]["Type"] = "State"
        i += 1
        
    for j in range(mu):
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

e["Variables"][i]["Name"] = "Cov Adaption"
e["Variables"][i]["Type"] = "Action"
e["Variables"][i]["Lower Bound"] = 0.0
e["Variables"][i]["Upper Bound"] = 1.0
e["Variables"][i]["Initial Exploration Noise"] = 0.2
i += 1


### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 10

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch"]["Size"] = 256

e["Solver"]["State Rescaling"]["Enabled"] = True
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

e["Solver"]["Termination Criteria"]["Max Experiences"] = maxExperiences

if evaluation == True:
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = list(range(repetitions))

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
