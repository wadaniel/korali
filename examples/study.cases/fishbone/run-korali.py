#!/usr/bin/env python3
import korali
import sys
import argparse
sys.path.append("_model/")
from model import *

k = korali.Engine()
e = korali.Experiment()

# Parsing arguments
parser = argparse.ArgumentParser(description='Runs the Aphros Fishbone experiment with Korali.')
parser.add_argument('--resultFolder', '-r', help='Path to the resuls folder', default='_result', required=False)
parser.add_argument('--samples', '-s', help='Number of CMAES samples per generation.', default=32, required=False)
parser.add_argument('--concurrency', '-c', help='Number of concurrent Aphros instances.', default=1, required=False)
parser.add_argument('--numCellsY', '-ny', help='Number of cells in the y-dimension.', default='64 * 6', required=False)
parser.add_argument('--numCores', '-nc', help='Number of cores per Aphros instance.', default=108, required=False)
parser.add_argument('--reynoldsNumber', '-re', help='Reynolds number for the simulation.', default=500, required=False)
parser.add_argument('--tmax', '-t', help='Maximum simulation time.', default=30, required=False)
parser.add_argument('--objective', '-obj', help='Column name of stat.dat to optimize.', default='vol2', required=False)
parser.add_argument('--ngens', '-ng', help='Number of generations to run per job.', default=1, required=False)
args = parser.parse_args()

# Parsing inputs
tmax = float(args.tmax)
objective = str(args.objective)
ngens = int(args.ngens)

# Loading previous results if they exist
resultFolder = args.resultFolder
e["File Output"]["Path"] = resultFolder
found = e.loadState(resultFolder + '/latest')

# If, found adding number of generations to the termination criteria
if (found == True):
 e["Solver"]["Termination Criteria"]["Max Generations"] = e["Current Generation"] + ngens
else:
 e["Solver"]["Termination Criteria"]["Max Generations"] = ngens
 
# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = lambda x: model(x, resultFolder, objective, tmax)

e["Variables"][0]["Name"] = "Bone Factor"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.8
e["Variables"][0]["Initial Standard Deviation"] = 0.9

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = args.samples
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-32

# Result Settings
e["File Output"]["Path"] = resultFolder
e["File Output"]["Frequency"] = 1 # Saving every state

# Selecting external conduit
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = int(args.concurrency)

# Reproducibility Settings
e["Random Seed"] = 0xC0FFEE
e["Preserve Random Number Generator States"] = True

# Configuring base parameter file
parameterString = ''
parameterString += 'np = ' + str(args.numCores) + '\n'
parameterString += 'ny = ' + str(args.numCellsY) + '\n'
parameterString += 'bubbles = True\n'
parameterString += 'tmax = ' + str(tmax) + '\n'
parameterString += 'Re = ' + str(args.reynoldsNumber) + '\n'

# Logging configuration
print('--------------------------------------------------')
print('Running Korali+Aphros Fishbone experiment.')
print('Result Folder: ' + resultFolder)
print('# Generations per job: ' + str(args.ngens))
print('CMAES samples per generation: ' + str(args.samples))
print('Concurrent Aphros instances: ' + str(args.concurrency))
print('Objective: ' + objective)
print('Base Configuration:')
print(parameterString, end='')
print('--------------------------------------------------')

# Storing base parameter file
configFile='_config/par.py'
with open(configFile, 'w') as f:
  print('[Korali] Creating: ' + configFile + '...')
  f.write(parameterString)
  
k.resume(e)
