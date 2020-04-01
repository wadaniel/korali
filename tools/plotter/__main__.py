#! /usr/bin/env python3
import os
import sys
import signal
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
import importlib

curdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Check if name has .png ending
def validateOutput(output):
    if not output.endswith(".png"):
        print("[Korali] Error: Outputfile '{0}' must end with '.png' suffix.".format(output))
        sys.exit(-1)

def main(path, mean, check, test, output):

 if (check == True):
  print("[Korali] Plotter correctly installed.")
  exit(0)

 if (test == True):
     matplotlib.use('Agg')

 signal.signal(signal.SIGINT, lambda x, y: exit(0))

 configFile = path + '/gen00000000.json'
 if ( not os.path.isfile(configFile) ):
  print("[Korali] Error: Did not find any results in the {0} folder...".format(path))
  exit(-1)

 with open(configFile) as f: js = json.load(f)
 configRunId = js['Run ID']
 
 resultFiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.startswith('gen')]
 resultFiles = sorted(resultFiles)
 
 genList = { } 
 
 for file in resultFiles:
  with open(path + '/' + file) as f:
   genJs = json.load(f)
   solverRunId = genJs['Run ID']
   
   if (configRunId == solverRunId):
     curGen = genJs['Current Generation']
     genList[curGen] = genJs

 del genList[0]
      
 requestedSolver = js['Solver']['Type']
 solverName = requestedSolver.rsplit('/')[-1]

 solverDir = curdir + '/../modules/solver/'
 for folder in requestedSolver.rsplit('/')[:-1]: solverDir += folder.lower()
 solverDir += '/' + solverName
 solverFile = solverDir + '/' + solverName + '.py'

 if os.path.isfile(solverFile):
  sys.path.append(solverDir)
  solverLib = importlib.import_module(solverName, package=None)
  fig, ax = solverLib.plot(genList)

 if not output:
  plt.show()
  exit(0)
 else:
  validateOutput(output)
  plt.savefig(output)
  exit(-1)

 if solverName == 'Executor':
    # TODO
    print("[Korali] No plotter for solver of type Executor available...")
    exit(0)

 if solverName == 'Rprop':
   # TODO
   print("[Korali] No plotter for solver of type Rprop available...")
   exit(0)

 if solverName == 'RecursiveDiscretizer':
   # TODO
   print("[Korali] No plotter for solver of type RecursiveDiscretizer available...")
   exit(0)
   
 print("[Korali] Error: Did not recognize solver '{0}' for plotting...".format(solverName))
 exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='korali.plotter', description='Plot the results of a Korali execution.')
    parser.add_argument('--dir', help='directory of result files', default='_korali_result', required = False)
    parser.add_argument('--mean', help='plot mean of objective variables', action='store_true', required = False)
    parser.add_argument('--check', help='verifies that korali.plotter is available', action='store_true', required = False)
    parser.add_argument('--test', help='run without graphics (for testing purpose)', action='store_true', required = False)
    parser.add_argument('--output', help='save figure to file', type=str, default="")
    args = parser.parse_args()

    main(args.dir, args.mean, args.check, args.test, args.output)
