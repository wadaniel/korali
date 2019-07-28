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
   solverHeaderString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
     
  for v in solverConfig["Termination Criteria"]:
   solverHeaderString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
   solverHeaderString += 'int ' + getVariableName(v) + 'Enabled;\n'
   solverHeaderString += 'int ' + getVariableName(v) + 'Triggered;\n'
   
  for v in solverConfig["Internal Settings"]:
   solverHeaderString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
       
  # Loading template header .hpp file
  solverTemplateHeaderFile = solverPath + '/' + solverName + '._hpp'
  with open(solverTemplateHeaderFile, 'r') as file: solverTemplateHeaderString = file.read()
  newHeaderString = solverTemplateHeaderString.replace('private:', 'private: \n' + solverHeaderString + '\n')
  
  # Saving new header .hpp file
  solverNewHeaderFile = solverPath + '/' + solverName + '.hpp'
  print('[Korali] Creating: ' + solverNewHeaderFile + '...')
  with open(solverNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Producing solver.cpp
  
  solverCodeString = 'void Korali::Solver::' + solverConfig["Alias"] + '::setConfiguration() \n{\n'
 
  # Consume Solver Settings
  for v in solverConfig["Solver Configuration"]:
    solverCodeString += consumeValue('_k->_js', solverConfig["Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [ 'Solver' ])
  
  for v in solverConfig["Internal Settings"]:
    solverCodeString += consumeValue('_k->_js', solverConfig["Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Internal' ])
  
  for v in solverConfig["Termination Criteria"]:
   terminationString = ''
   
   if (not getVariableDefault(v)): 
    solverCodeString += '\n ' + getVariableName(v) + 'Enabled = false;'
    terminationString = consumeValue('_k->_js', solverConfig["Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Termination Criteria' ])
    terminationString = terminationString.replace(getVariableName(v) + ' = ', getVariableName(v) + 'Enabled = true;\n  ' + getVariableName(v) + ' = ') 
   else:
    solverCodeString += '\n ' + getVariableName(v) + 'Enabled = true;'
    terminationString = consumeValue('_k->_js', solverConfig["Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [ 'Solver', 'Termination Criteria' ])
  
   solverCodeString += terminationString
   solverCodeString += consumeValue('_k->_js', solverConfig["Alias"], v["Name"] + ' Triggered', getVariableName(v) + 'Triggered', 'int', 'false', [ 'Solver', 'Termination Criteria' ])  
 
  solverCodeString += '} \n\n'
  
  ###### Creating Solver Get Configuration routine
  
  solverCodeString += 'void Korali::Solver::' + solverConfig["Alias"]  + '::getConfiguration() \n{\n\n'
  solverCodeString += ' _k->_js["Solver"]["Type"] = "' + solverConfig["Alias"] + '";\n'
 
  for v in solverConfig["Solver Configuration"]: 
    solverCodeString += ' _k->_js["Solver"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
    
  for v in solverConfig["Internal Settings"]: 
    solverCodeString += ' _k->_js["Solver"]["Internal"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
  
  for v in solverConfig["Termination Criteria"]: 
    solverCodeString += ' if (' + getVariableName(v) + 'Enabled == 1) _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
    solverCodeString += ' _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + ' Triggered"] = ' + getVariableName(v) + 'Triggered;\n'
  
  solverCodeString += '} \n\n'
  
  ###### Creating Solver Check Termination routine
  
  solverCodeString += 'bool Korali::Solver::' + solverConfig["Alias"]  + '::checkTermination()\n'
  solverCodeString += '{\n'
  solverCodeString += ' bool hasFinished = false;\n\n'
 
  for v in solverConfig["Termination Criteria"]: 
    solverCodeString += ' if (' + getVariableName(v) + 'Enabled == true)\n'
    solverCodeString += ' if (' + v["Criteria"] + ')\n'
    solverCodeString += ' {\n'
    solverCodeString += '  ' + getVariableName(v) + 'Triggered = true;\n'
    solverCodeString += '  koraliLog(KORALI_MINIMAL, "' + solverConfig["Alias"] + ' Termination Criteria met: \\"' + v["Name"] + '\\" (%s).\\n", std::to_string(' + getVariableName(v)  +').c_str());\n'
    solverCodeString += '  hasFinished = true;\n'
    solverCodeString += ' }\n\n'
  
  solverCodeString += ' return hasFinished;\n'
  solverCodeString += '}'
  
  ###### Creating code file
  with open(solverPath + '/' + solverName + '._cpp', 'r') as file: solverBaseCodeString = file.read()
  solverBaseCodeString += '\n\n' + solverCodeString
  solverNewCodeFile = solverPath + '/' + solverName + '.cpp'
  print('[Korali] Creating: ' + solverNewCodeFile + '...')
  with open(solverNewCodeFile, 'w') as file: file.write(solverBaseCodeString)
