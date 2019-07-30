import sys
import os
import json
from buildAux import *

def buildVariables(koraliDir):
  # Reading base variable file
 with open(koraliDir + '/variable/variable._cpp', 'r') as file: variableCodeString = file.read()
 
 # Creating variable settings
 variableSettingDeclarationsString = '' 
 variableSettingSet = set()
 
 ########  Detecting Solvers
 
 # Creating External Setting Strings for Variables
 variableSetSolverSettingString = ''
 variableGetSolverSettingString = ''
 
 solverdir = koraliDir + '/solvers' 
 solverPaths  = [x[0] for x in os.walk(solverdir)][1:]
 for solverPath in solverPaths:
  solverName = solverPath.replace(solverdir + '/', '')
  
  # Loading JSON Configuration
  solverJsonFile = solverPath + '/' + solverName + '.json'
  if (not os.path.isfile(solverJsonFile)): continue 
  with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
  solverConfig = json.loads(solverJsonString)
  
  # Adding information to variables.hpp
  
  # Reading variable-specific configuration
  
  for v in solverConfig["Variables Configuration"]:
   if (not getVariablePath(v) in variableSettingSet):
    variableSettingDeclarationsString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
    variableSettingSet.add(getVariablePath(v)) 
    
  # Creating set variable configuration
  
  variableSetSolverSettingString += ' if (_k->_solverType == "' + solverConfig["Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableSetSolverSettingString += consumeValue('js', solverConfig["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
  variableSetSolverSettingString += ' }\n'
  
  # Creating get variable configuration
  
  variableGetSolverSettingString += ' if (_k->_solverType == "' + solverConfig["Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableGetSolverSettingString += '  js' + getVariablePath(v) + ' = ' + getCXXVariableName(v) + ';\n'
  variableGetSolverSettingString += ' }\n\n'
 
 # Saving variable solver configuration
 variableCodeString += 'void Korali::Variable::setSolverSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableSetSolverSettingString
 variableCodeString += '}\n\n'
 
 # Saving variable solver configuration
 variableCodeString += 'void Korali::Variable::getSolverSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableGetSolverSettingString
 variableCodeString += '}\n\n'
 
 ########  Detecting Problems
 
 # Creating External Setting Strings for Variables
 variableSetProblemSettingString = ''
 variableGetProblemSettingString = ''
 
 problemdir = koraliDir + '/problems'
 problemPaths  = [x[0] for x in os.walk(problemdir)][1:]
 for problemPath in problemPaths:
  problemName = problemPath.replace(problemdir + '/', '')
  
  # Loading JSON Configuration
  problemJsonFile = problemPath + '/' + problemName + '.json'
  if (not os.path.isfile(problemJsonFile)): continue 
  with open(problemJsonFile, 'r') as file: problemJsonString = file.read()
  problemConfig = json.loads(problemJsonString)
  
  # Adding information to variables.hpp
  
  # Reading variable-specific configuration
  
  for v in problemConfig["Variables Configuration"]:
   if (not getVariablePath(v) in variableSettingSet):
    variableSettingDeclarationsString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
    variableSettingSet.add(getVariablePath(v))
    
  # Creating set variable configuration
  
  variableSetProblemSettingString += ' if ( _k->_problemType == "' + problemConfig["Alias"] + '")\n {\n'
  for v in problemConfig["Variables Configuration"]: 
    variableSetProblemSettingString += consumeValue('js', problemConfig["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
  variableSetProblemSettingString += ' }\n'
  
  # Creating get variable configuration
  
  variableGetProblemSettingString += ' if ( _k->_problemType == "' + problemConfig["Alias"] + '")\n {\n'
  for v in problemConfig["Variables Configuration"]: 
    variableGetProblemSettingString += saveValue('_k->_js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
  variableGetProblemSettingString += ' }\n\n'
 
 # Saving set variable problem configuration
 variableCodeString += 'void Korali::Variable::setProblemSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableSetProblemSettingString
 variableCodeString += '}\n\n'
 
 # Saving get variable problem configuration
 variableCodeString += 'void Korali::Variable::getProblemSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableGetProblemSettingString
 variableCodeString += '}\n\n'
 
 ###### Finished Parsing Problems
  
 # Saving new variables.hpp file
 with open(koraliDir + '/variable/variable._hpp', 'r') as file: variableHeaderString = file.read()
 variableNewHeaderFile = koraliDir + '/variable/variable.hpp'
 newHeaderString = variableHeaderString.replace('public:', 'public: \n' + variableSettingDeclarationsString + '\n')
 print('[Korali] Creating: ' + variableNewHeaderFile + '...')
 with open(variableNewHeaderFile, 'w') as file: file.write(newHeaderString)
 
 variableNewCodeFile = koraliDir + '/variable/variable.cpp'
 print('[Korali] Creating: ' + variableNewCodeFile + '...')
 with open(variableNewCodeFile, 'w') as file: file.write(variableCodeString)
 
 
 
 
