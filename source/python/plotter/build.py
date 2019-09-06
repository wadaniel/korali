#!/usr/bin/env python3

import os
import sys
from shutil import copyfile

currentDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
koraliDir = currentDir + '/../..' 

print('[Korali] Building plotter script...')
with open(currentDir + '/__main__._py', 'r') as file: plotterString = file.read()
 
importsString = ''
detectString = ''
 
solversDir = koraliDir + '/modules/solvers'
solverPaths  = [x[0] for x in os.walk(solversDir)][1:]
for solverPath in solverPaths:
 solverName = solverPath.replace(solversDir + '/', '')
 solverPythonFile = solverPath + '/' + solverName + '.py'
 if (not os.path.isfile(solverPythonFile)): continue 
 
 importsString += " from korali.plotter." + solverName + " import plot_" + solverName + "\n"
 copyfile(solverPythonFile, koraliDir + '/python/plotter/' + solverName + '.py')
   
 detectString += " if ( '" + solverName.upper() + "' == solver):\n"
 detectString += '  print("[Korali] Running ' + solverName.upper() + ' Plotter...")\n'
 detectString += '  plot_' + solverName + '(path, allFiles, live, generation, test, mean)\n'
 detectString += '  exit(0)\n\n'
  
plotterString = plotterString.replace('# Including Solvers', importsString)
plotterString = plotterString.replace(' # Detecting Solver Type', detectString)
 
with open(currentDir + '/__main__.py', 'w') as file: file.write(plotterString)
