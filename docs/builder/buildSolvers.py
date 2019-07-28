import sys
import os
import json
from buildAux import *

def buildSolvers(koraliDir):
 # Processing Solvers
 curdir = koraliDir + '/solvers' 
 
 # Detecting Solvers
 solverPaths  = [x[0] for x in os.walk(curdir)][1:]
 for solverPath in solverPaths:
  solverName = solverPath.replace(curdir + '/', '')
  
  # Loading JSON Configuration
  solverJsonFile = solverPath + '/' + solverName + '.json'
  if (not os.path.isfile(solverJsonFile)): continue 
  with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
  solverConfig = json.loads(solverJsonString)
  
  print('Producing ' + solverName + '...')
  
  ####### Producing solver page

  solverDocFile = solverPath + '/' + solverName + '.md'
  with open(solverDocFile, 'r') as file: solverDocString = file.read()
  
  solverSettingsString = ''
  
  solverSettingsString += '### Solver Settings\n\n'
  for v in solverConfig["Module Configuration"]:
   solverSettingsString += getVariableInfo(v, solverName)
    
  solverSettingsString += '### Variable-Specific Settings\n\n'
  for v in solverConfig["Variables Configuration"]: 
    solverSettingsString += getVariableInfo(v, solverName)
    
  solverSettingsString += '### Termination Criteria\n\n'
  for v in solverConfig["Termination Criteria"]: 
    solverSettingsString += getVariableInfo(v, solverName)
  
  solverDocString = solverDocString.replace('## Configuration', '## Configuration\n\n' + solverSettingsString + '\n\n') 
    
  with open('../docs/usage/solvers/' + solverName + '.md', 'w+') as file: file.write(solverDocString)
