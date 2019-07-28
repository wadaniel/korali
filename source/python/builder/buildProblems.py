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
   problemHeaderString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
     
  for v in problemConfig["Internal Settings"]:
   problemHeaderString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
       
  # Loading template header .hpp file
  problemTemplateHeaderFile = problemPath + '/' + problemName + '._hpp'
  with open(problemTemplateHeaderFile, 'r') as file: problemTemplateHeaderString = file.read()
  newHeaderString = problemTemplateHeaderString.replace('private:', 'private: \n' + problemHeaderString + '\n')
  
  # Saving new header .hpp file
  problemNewHeaderFile = problemPath + '/' + problemName + '.hpp'
  print('[Korali] Creating: ' + problemNewHeaderFile + '...')
  with open(problemNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Producing problem.cpp
  
  problemCodeString = 'void Korali::Problem::' + problemConfig["Alias"] + '::setConfiguration() \n{\n'
 
  # Consume Problem Settings
  for v in problemConfig["Problem Configuration"]:
    problemCodeString += consumeValue('_k->_js', problemConfig["Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [ 'Problem' ])
  
  for v in problemConfig["Internal Settings"]:
    problemCodeString += consumeValue('_k->_js', problemConfig["Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Problem', 'Internal' ])
  
  problemCodeString += '} \n\n'
  
  ###### Creating Problem Get Configuration routine
  
  problemCodeString += 'void Korali::Problem::' + problemConfig["Alias"]  + '::getConfiguration() \n{\n\n'
  problemCodeString += ' _k->_js["Problem"]["Type"] = "' + problemConfig["Alias"] + '";\n'
 
  for v in problemConfig["Problem Configuration"]: 
    problemCodeString += ' _k->_js["Problem"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
    
  for v in problemConfig["Internal Settings"]: 
    problemCodeString += ' _k->_js["Problem"]["Internal"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
  
  problemCodeString += '} \n\n'
  
  ###### Creating code file
  with open(problemPath + '/' + problemName + '._cpp', 'r') as file: problemBaseCodeString = file.read()
  problemBaseCodeString += '\n\n' + problemCodeString
  problemNewCodeFile = problemPath + '/' + problemName + '.cpp'
  print('[Korali] Creating: ' + problemNewCodeFile + '...')
  with open(problemNewCodeFile, 'w') as file: file.write(problemBaseCodeString)
