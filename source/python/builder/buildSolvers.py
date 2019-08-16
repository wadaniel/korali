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
  
  for v in solverConfig["Solver Configuration"]:
   solverHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
     
  for v in solverConfig["Termination Criteria"]:
   solverHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
   solverHeaderString += 'int ' + getCXXVariableName(v) + 'Enabled;\n'
   solverHeaderString += 'int ' + getCXXVariableName(v) + 'Triggered;\n'
   
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
  
  ###### Producing solver.cpp
  
  solverCodeString = 'void Korali::Solver::' + solverConfig["Class"] + '::setConfiguration(nlohmann::json& js) \n{\n'
 
  # Consume Solver Settings
  for v in solverConfig["Solver Configuration"]:
    solverCodeString += consumeValue('js', solverConfig["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
  
  for v in solverConfig["Internal Settings"]:
    solverCodeString += consumeValue('js', solverConfig["Alias"], '["Internal"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v), 'Korali Skip Default')
  
  for v in solverConfig["Termination Criteria"]:
    solverCodeString += consumeValue('js', solverConfig["Alias"], '["Termination Criteria"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
 
  solverCodeString += ' if(isEmpty(js) == false) Korali::logError("Unrecognized settings for the ' + solverConfig["Name"] + ' solver: \\n%s\\n", js.dump(2).c_str());\n'
  solverCodeString += '} \n\n'
   
  ###### Creating Solver Get Configuration routine
    
  solverCodeString += 'void Korali::Solver::' + solverConfig["Class"]  + '::getConfiguration(nlohmann::json& js) \n{\n\n' 
  
  for v in solverConfig["Solver Configuration"]: 
    solverCodeString += saveValue('js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
    
  for v in solverConfig["Internal Settings"]: 
    solverCodeString += saveValue('js', '["Internal"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v))
  
  for v in solverConfig["Termination Criteria"]: 
    solverCodeString += saveValue('js', '["Termination Criteria"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v))
  
  solverCodeString += '} \n\n'
  
  ###### Creating Solver Check Termination routine
  
  solverCodeString += 'bool Korali::Solver::' + solverConfig["Class"]  + '::checkTermination()\n'
  solverCodeString += '{\n'
  solverCodeString += ' bool hasFinished = false;\n\n'
 
  for v in solverConfig["Termination Criteria"]: 
    solverCodeString += ' if (' + v["Criteria"] + ')\n'
    solverCodeString += ' {\n'
    solverCodeString += '  Korali::logInfo("Minimal", "' + solverConfig["Alias"] + ' Termination Criteria met: \\"' + getVariablePath(v).replace('"', "'") + '\\" (' + getVariableDescriptor(v) + ').\\n", ' + getCXXVariableName(v)  +');\n'
    solverCodeString += '  hasFinished = true;\n'
    solverCodeString += ' }\n\n'
  
  solverCodeString += ' return hasFinished;\n'
  solverCodeString += '}'
  
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
