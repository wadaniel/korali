#!/usr/bin/env python3

import os
import sys
from shutil import copyfile

def buildPlotter(koraliDir):
 print('[Korali] Building plotter script...')
 with open(koraliDir + '/python/plotter/__main__._py', 'r') as file: plotterString = file.read()
 
 importsString = ''
 detectString = ''
 
 solversDir = koraliDir + '/solvers'
 solverPaths  = [x[0] for x in os.walk(solversDir)][1:]
 for solverPath in solverPaths:
   solverName = solverPath.replace(solversDir + '/', '')
   solverPythonFile = solverPath + '/' + solverName + '.py'
   if (not os.path.isfile(solverPythonFile)): continue 
   #print(solverPath)
   
   importsString += " from korali.plotter." + solverName + " import plot_" + solverName + "\n"
   copyfile(solverPythonFile, koraliDir + '/python/plotter/' + solverName + '.py')
   
   detectString += " if ( '" + solverName.upper() + "' == solver):\n"
   detectString += '  print("[Korali] Running ' + solverName.upper() + ' Plotter...")\n'
   detectString += '  plot_' + solverName + '(path, allFiles, live, generation, test, mean)\n'
   detectString += '  exit(0)\n\n'
   
   #with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
   #solverConfig = json.loads(solverJsonString)
  
 plotterString = plotterString.replace('# Including Solvers', importsString)
 plotterString = plotterString.replace(' # Detecting Solver Type', detectString)
 
 with open(koraliDir + '/python/plotter/__main__.py', 'w') as file: file.write(plotterString)
