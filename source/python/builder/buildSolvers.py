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
  
  ####### Producing solver.hpp
  
  # Producing private variable declarations
  solverHeaderString = ''
  
  for v in solverConfig["Configuration Settings"]:
   solverHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
     
  for v in solverConfig["Termination Criteria"]:
   solverHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
   
  for v in solverConfig["Internal Settings"]:
   solverHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
       
  # Loading template header .hpp file
  solverTemplateHeaderFile = solverPath + '/' + solverName + '._hpp'
  with open(solverTemplateHeaderFile, 'r') as file: solverTemplateHeaderString = file.read()
  newHeaderString = solverTemplateHeaderString.replace('private:', 'private: \n' + solverHeaderString + '\n')
  
  # Saving new header .hpp file
  solverNewHeaderFile = solverPath + '/' + solverName + '.hpp'
  print('[Korali] Creating: ' + solverNewHeaderFile + '...')
  with open(solverNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Producing solver code
  
  solverCodeString = ''  
  solverCodeString += createSetConfiguration(solverConfig)
  solverCodeString += createGetConfiguration(solverConfig)
  solverCodeString += createCheckTermination(solverConfig)
  
  ###### Creating code file
  
  solverBaseFileName = solverPath + '/' + solverName + '._cpp'
  solverNewCodeFile = solverPath + '/' + solverName + '.cpp'
  baseFileTime = os.path.getmtime(solverBaseFileName)
  newFileTime = baseFileTime
  if (os.path.exists(solverNewCodeFile)): newFileTime = os.path.getmtime(solverNewCodeFile)
  
  if (baseFileTime >= newFileTime):
    with open(solverBaseFileName, 'r') as file: solverBaseCodeString = file.read()
    solverBaseCodeString += '\n\n' + solverCodeString
    print('[Korali] Creating: ' + solverNewCodeFile + '...')
    with open(solverNewCodeFile, 'w') as file: file.write(solverBaseCodeString)
