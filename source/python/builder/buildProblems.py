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
  
  for v in problemConfig["Problem Configuration"]:
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
  
  ###### Producing problem.cpp
  
  problemCodeString = 'void Korali::Problem::' + problemConfig["Class"] + '::setConfiguration(nlohmann::json& js) \n{\n'
  
  # Checking whether solver is accepted
  problemCodeString += ' bool __acceptedSolver = false;\n'
  for v in problemConfig["Compatible Solvers"]:
   problemCodeString += ' if (_k->_solverType == "' + v + '") __acceptedSolver = true;\n'
  problemCodeString += ' if (__acceptedSolver == false) Korali::logError("Selected solver %s not compatible with problem type ' + problemConfig["Name"] + '", _k->_solverType.c_str()); \n\n' 
 
  # Consume Problem Settings
  for v in problemConfig["Problem Configuration"]:
    problemCodeString += consumeValue('js', problemConfig["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
  
  for v in problemConfig["Internal Settings"]:
    problemCodeString += consumeValue('js', problemConfig["Alias"], '["Internal"]' + getVariablePath(v),  getCXXVariableName(v), getVariableType(v), 'Korali Skip Default')
  
  problemCodeString += ' if(isEmpty(js) == false) Korali::logError("Unrecognized settings for the ' + problemConfig["Name"] + ' problem: \\n%s\\n", js.dump(2).c_str());\n'
  problemCodeString += '} \n\n'
  
  ###### Creating Problem Get Configuration routine
  
  problemCodeString += 'void Korali::Problem::' + problemConfig["Class"]  + '::getConfiguration(nlohmann::json& js) \n{\n\n'
 
  for v in problemConfig["Problem Configuration"]: 
    problemCodeString += saveValue('js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
    
  for v in problemConfig["Internal Settings"]: 
    problemCodeString += saveValue('js', '["Internal"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v))
  
  problemCodeString += '} \n\n'
  
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
