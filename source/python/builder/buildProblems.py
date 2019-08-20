import sys
import os
import json
from buildAux import *

def buildProblems(koraliDir):
 # Processing Problems
 curdir = koraliDir + '/problems' 
 
 # Detecting Problems
 problemPaths  = [x[0] for x in os.walk(curdir)][1:]
 for problemPath in problemPaths:
  problemName = problemPath.replace(curdir + '/', '')
  
  # Loading JSON Configuration
  problemJsonFile = problemPath + '/' + problemName + '.json'
  if (not os.path.isfile(problemJsonFile)): continue 
  with open(problemJsonFile, 'r') as file: problemJsonString = file.read()
  problemConfig = json.loads(problemJsonString)
  
  ####### Producing problem.hpp
  
  # Producing private variable declarations
  
  problemHeaderString = ''
  
  for v in problemConfig["Configuration Settings"]:
   problemHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
     
  for v in problemConfig["Internal Settings"]:
   problemHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
       
  # Loading template header .hpp file
  problemTemplateHeaderFile = problemPath + '/' + problemName + '._hpp'
  with open(problemTemplateHeaderFile, 'r') as file: problemTemplateHeaderString = file.read()
  newHeaderString = problemTemplateHeaderString.replace('private:', 'private: \n' + problemHeaderString + '\n')
  
  # Saving new header .hpp file
  problemNewHeaderFile = problemPath + '/' + problemName + '.hpp'
  print('[Korali] Creating: ' + problemNewHeaderFile + '...')
  with open(problemNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Producing problem code

  problemCodeString = ''  
  problemCodeString += createSetConfiguration(problemConfig)
  problemCodeString += createGetConfiguration(problemConfig)
  problemCodeString += createCheckTermination(problemConfig)
  
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
