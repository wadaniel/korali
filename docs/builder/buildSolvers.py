import sys
import os
import json
from buildAux import *

def buildSolvers(koraliDir):
 # Processing Solvers
 solversDir = koraliDir + '/source/solvers'
 
 # Initializing Mkdocs entries string
 mkdocsEntriesString = ''
  
 # Detecting Solvers
 solverPaths  = [x[0] for x in os.walk(solversDir)][1:]
 for solverPath in solverPaths:
  solverName = solverPath.replace(solversDir + '/', '')
  
  # Loading JSON Configuration
  solverJsonFile = solverPath + '/' + solverName + '.json'
  if (not os.path.isfile(solverJsonFile)): continue 
  with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
  solverConfig = json.loads(solverJsonString)
  
  # Adding yaml entry
  mkdocsEntriesString += '          - ' + solverConfig["Alias"] + ': usage/solvers/' + solverName + '.md\n'
  
  ####### Producing solver page

  solverDocFile = solverPath + '/' + solverName + '.md'
  with open(solverDocFile, 'r') as file: solverDocString = file.read()
  
  solverDocString = '# ' + solverConfig["Name"] + '\n\n' + solverDocString 
  
  solverSettingsString = ''
  
  solverSettingsString += '### Solver Settings\n\n'
  for v in solverConfig["Solver Configuration"]:
   solverSettingsString += getVariableInfo(v, solverName)
    
  solverSettingsString += '### Variable-Specific Settings\n\n'
  for v in solverConfig["Variables Configuration"]: 
    solverSettingsString += getVariableInfo(v, solverName)
    
  solverSettingsString += '### Termination Criteria\n\n'
  for v in solverConfig["Termination Criteria"]: 
    solverSettingsString += getVariableInfo(v, solverName)
  
  solverDocString = solverDocString.replace('## Configuration', '## Configuration\n\n' + solverSettingsString + '\n\n') 

  mdFileName = koraliDir + '/docs/docs/usage/solvers/' + solverName + '.md'
  print('[Korali] Creating ' + mdFileName + '...')    
  with open(mdFileName, 'w+') as file: file.write(solverDocString)

 return mkdocsEntriesString