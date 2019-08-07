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
 
 # Creating External Setting Strings for Variables
 variableSetSettingString = ''
 variableGetSettingString = ''
 
 ########  Detecting Solvers
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
    variableGetSettingString += ' bool is' + getCXXVariableName(v) + 'Defined = false;\n'
    variableSetSettingString += ' bool is' + getCXXVariableName(v) + 'Defined = false;\n'
    variableSettingDeclarationsString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
    variableSettingSet.add(getVariablePath(v)) 
    
  # Creating set variable configuration
  
  variableSetSettingString += ' if (_k->_solverType == "' + solverConfig["Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableSetSettingString += ' if (is' + getCXXVariableName(v) + 'Defined == false)'
    variableSetSettingString += ' {\n'
    variableSetSettingString += consumeValue('js', solverConfig["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
    variableSetSettingString += '  is' + getCXXVariableName(v) + 'Defined = true;\n\n'
    variableSetSettingString += ' }\n'
    
  variableSetSettingString += ' }\n\n'
  
  # Creating get variable configuration
  
  variableGetSettingString += ' if (_k->_solverType == "' + solverConfig["Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableGetSettingString += saveValue('js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
    
  variableGetSettingString += ' }\n\n'
 
 ########  Detecting Problems
 
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
    variableGetSettingString += ' bool is' + getCXXVariableName(v) + 'Defined = false;\n'
    variableSetSettingString += ' bool is' + getCXXVariableName(v) + 'Defined = false;\n'
    variableSettingDeclarationsString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
    variableSettingSet.add(getVariablePath(v))
    
  # Creating set variable configuration
  
  variableSetSettingString += ' if ( _k->_problemType == "' + problemConfig["Alias"] + '")\n {\n'
  for v in problemConfig["Variables Configuration"]: 
    variableSetSettingString += ' if (is' + getCXXVariableName(v) + 'Defined == false)'
    variableSetSettingString += ' {\n'
    variableSetSettingString += consumeValue('js', problemConfig["Alias"], getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
    variableSetSettingString += '  is' + getCXXVariableName(v) + 'Defined = true;\n'
    variableSetSettingString += ' }\n\n'
    
  variableSetSettingString += ' }\n\n'
  
  # Creating get variable configuration
  
  variableGetSettingString += ' if ( _k->_problemType == "' + problemConfig["Alias"] + '")\n {\n'
  for v in problemConfig["Variables Configuration"]: 
    variableGetSettingString += saveValue('js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
    
  variableGetSettingString += ' }\n\n'
 
 ###### Saving variable inherited set configuration
 
 variableCodeString += 'void Korali::Variable::setInheritedSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableSetSettingString
 variableCodeString += '}\n\n'
 
 ###### Saving variable inherited get configuration
 variableCodeString += 'void Korali::Variable::getInheritedSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableGetSettingString
 variableCodeString += '}\n\n'
 
 ###### Finished Parsing Problems
  
 # Saving new variables.hpp file
 with open(koraliDir + '/variable/variable._hpp', 'r') as file: variableHeaderString = file.read()
 variableNewHeaderFile = koraliDir + '/variable/variable.hpp'
 newHeaderString = variableHeaderString.replace('public:', 'public: \n' + variableSettingDeclarationsString + '\n')
 print('[Korali] Creating: ' + variableNewHeaderFile + '...')
 with open(variableNewHeaderFile, 'w') as file: file.write(newHeaderString)

 variableNewCodeFile = koraliDir + '/variable/variable.cpp' 
 if (not os.path.exists(variableNewCodeFile)):
  print('[Korali] Creating: ' + variableNewCodeFile + '...')
  with open(variableNewCodeFile, 'w') as file: file.write(variableCodeString)
 
 
 
 
