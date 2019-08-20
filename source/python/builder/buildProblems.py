import sys
import os
import json
from buildAux import *

def buildProblems(koraliDir):
 # Processing Problems
 curdir = koraliDir + '/problems' 
 
 # Problems List
 problemDetectionList = ''
 
 # Detecting Problems
 problemPaths  = [x[0] for x in os.walk(curdir)][1:]
 for problemPath in problemPaths:
  problemName = problemPath.replace(curdir + '/', '')
  
  # Loading JSON Configuration
  problemJsonFile = problemPath + '/' + problemName + '.json'
  if (not os.path.isfile(problemJsonFile)): continue 
  with open(problemJsonFile, 'r') as file: problemJsonString = file.read()
  problemConfig = json.loads(problemJsonString)
  
  ####### Adding problem to list

  problemDetectionList += '  if(problemType == "' + problemConfig["Alias"] + '") problem = new Korali::Problem::' + problemConfig["C++ Class"] + '();\n'
  
  ###### Producing problem code

  problemHeaderString = createHeaderDeclarations(problemConfig)
  problemCodeString = createSetConfiguration(problemConfig)
  problemCodeString += createGetConfiguration(problemConfig)
  problemCodeString += createCheckTermination(problemConfig)

  ####### Producing header file
  
  # Loading template header .hpp file
  problemTemplateHeaderFile = problemPath + '/' + problemName + '._hpp'
  with open(problemTemplateHeaderFile, 'r') as file: problemTemplateHeaderString = file.read()
  newHeaderString = problemTemplateHeaderString.replace('private:', 'private: \n' + problemHeaderString + '\n')
  
  # Saving new header .hpp file
  problemNewHeaderFile = problemPath + '/' + problemName + '.hpp'
  print('[Korali] Creating: ' + problemNewHeaderFile + '...')
  with open(problemNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Creating code file
  
  problemBaseFileName = problemPath + '/' + problemName + '._cpp'
  problemNewCodeFile = problemPath + '/' + problemName + '.cpp'
  baseFileTime = os.path.getmtime(problemBaseFileName)
  newFileTime = baseFileTime
  if (os.path.exists(problemNewCodeFile)): newFileTime = os.path.getmtime(problemNewCodeFile)
  
  if (baseFileTime >= newFileTime):
   with open(problemBaseFileName, 'r') as file: problemBaseCodeString = file.read()
   problemBaseCodeString += '\n\n' + problemCodeString
   print('[Korali] Creating: ' + problemNewCodeFile + '...')
   with open(problemNewCodeFile, 'w') as file: file.write(problemBaseCodeString)

 ###### Creating base configuration file

 problemBaseFileName = curdir + '/base._cpp'
 problemNewCodeFile = curdir + '/base.cpp'
 baseFileTime = os.path.getmtime(problemBaseFileName)
 newFileTime = baseFileTime
 if (os.path.exists(problemNewCodeFile)): newFileTime = os.path.getmtime(problemNewCodeFile)

 if (baseFileTime >= newFileTime):
   with open(problemBaseFileName, 'r') as file: problemBaseCodeString = file.read()
   newBaseString = problemBaseCodeString.replace(' // Problem list', problemDetectionList)
   with open(problemNewCodeFile, 'w') as file: file.write(newBaseString)

 ###### Creating base header file
 with open(curdir + '/base._hpp', 'r') as file: problemBaseHeaderString = file.read()
 newBaseString = problemBaseHeaderString
 with open(curdir + '/base.hpp', 'w+') as file: file.write(newBaseString)