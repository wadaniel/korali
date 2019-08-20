import sys
import os
import json
from buildAux import *

def buildSolvers(koraliDir):
 # Processing Solvers
 curdir = koraliDir + '/solvers' 
 
 # Solvers List
 solverDetectionList = ''
 
 # Detecting Solvers
 solverPaths  = [x[0] for x in os.walk(curdir)][1:]
 for solverPath in solverPaths:
  solverName = solverPath.replace(curdir + '/', '')
  
  # Loading JSON Configuration
  solverJsonFile = solverPath + '/' + solverName + '.json'
  if (not os.path.isfile(solverJsonFile)): continue 
  with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
  solverConfig = json.loads(solverJsonString)
  
  ####### Adding solver to list

  solverDetectionList += '  if(solverType == "' + solverConfig["Alias"] + '") solver = new Korali::Solver::' + solverConfig["C++ Class"] + '();\n'
  
  ###### Producing solver code

  solverHeaderString = createHeaderDeclarations(solverConfig)
  solverCodeString = createSetConfiguration(solverConfig)
  solverCodeString += createGetConfiguration(solverConfig)
  solverCodeString += createCheckTermination(solverConfig)

  ####### Producing header file
  
  # Loading template header .hpp file
  solverTemplateHeaderFile = solverPath + '/' + solverName + '._hpp'
  with open(solverTemplateHeaderFile, 'r') as file: solverTemplateHeaderString = file.read()
  newHeaderString = solverTemplateHeaderString.replace('private:', 'private: \n' + solverHeaderString + '\n')
  
  # Saving new header .hpp file
  solverNewHeaderFile = solverPath + '/' + solverName + '.hpp'
  print('[Korali] Creating: ' + solverNewHeaderFile + '...')
  with open(solverNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
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

 ###### Creating base configuration file

 solverBaseFileName = curdir + '/base._cpp'
 solverNewCodeFile = curdir + '/base.cpp'
 baseFileTime = os.path.getmtime(solverBaseFileName)
 newFileTime = baseFileTime
 if (os.path.exists(solverNewCodeFile)): newFileTime = os.path.getmtime(solverNewCodeFile)

 if (baseFileTime >= newFileTime):
   with open(solverBaseFileName, 'r') as file: solverBaseCodeString = file.read()
   newBaseString = solverBaseCodeString.replace(' // Solver list', solverDetectionList)
   with open(solverNewCodeFile, 'w') as file: file.write(newBaseString)

 ###### Creating base header file
 with open(curdir + '/base._hpp', 'r') as file: solverBaseHeaderString = file.read()
 newBaseString = solverBaseHeaderString
 with open(curdir + '/base.hpp', 'w+') as file: file.write(newBaseString)