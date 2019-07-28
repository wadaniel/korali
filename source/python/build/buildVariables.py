import sys
import os
import json
from buildAux import *

def buildVariables(koraliDir):
 # Processing Solvers
 solverdir = koraliDir + '/solvers' 
 
 # Creating variable settings
 variableSettingDeclarationsString = '' 
 variableSettingSet = set()
 
 # Creating External Setting Strings for Variables
 variableSetSolverSettingString = ''
 variableGetSolverSettingString = ''
 
 # Detecting Solvers
 solverPaths  = [x[0] for x in os.walk(solverdir)][1:]
 for solverPath in solverPaths:
  solverName = solverPath.replace(solverdir + '/', '')
  
  # Loading JSON Configuration
  solverJsonFile = solverPath + '/' + solverName + '.json'
  if (not os.path.isfile(solverJsonFile)): continue 
  with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
  solverConfig = json.loads(solverJsonString)
  
  ####### Adding information to variables.hpp
  
  # Reading variable-specific configuration
  
  for v in solverConfig["Variables Configuration"]:
   if (not v["Name"] in variableSettingSet):
    variableSettingDeclarationsString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
    variableSettingSet.add(v["Name"])
    
  # Creating set variable configuration
  
  variableSetSolverSettingString += ' if (isDefined(_k->_js, { "Solver", "Type" })) if (_k->_js["Solver"]["Type"] == "' + solverConfig["Module Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableSetSolverSettingString += consumeValue('js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [  ])
  variableSetSolverSettingString += ' }\n'
  
  # Creating get variable configuration
  
  variableGetSolverSettingString += ' if (isDefined(_k->_js, { "Solver", "Type" })) if (_k->_js["Solver"]["Type"] == "' + solverConfig["Module Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableGetSolverSettingString += '  js["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
  variableGetSolverSettingString += ' }\n\n'
 
 ###### Finished Parsing Solvers
  
 # Saving new variables.hpp file
 with open(koraliDir + '/variable._hpp', 'r') as file: variableHeaderString = file.read()
 variableNewHeaderFile = koraliDir + '/variable.hpp'
 newHeaderString = variableHeaderString.replace('public:', 'public: \n' + variableSettingDeclarationsString + '\n')
 print('[Korali] Creating: ' + variableNewHeaderFile + '...')
 with open(variableNewHeaderFile, 'w') as file: file.write(newHeaderString)
 
 # Saving new config.cpp file
 with open(koraliDir + '/variable._cpp', 'r') as file: variableCodeString = file.read()
 
 # Saving variable solver configuration
 variableCodeString += 'void Korali::Variable::setSolverSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableSetSolverSettingString
 variableCodeString += '}\n\n'
 
 # Saving variable solver configuration
 variableCodeString += 'void Korali::Variable::getSolverSettings(nlohmann::json& js)\n{\n'
 variableCodeString += variableGetSolverSettingString
 variableCodeString += '}\n\n'
 
 variableNewCodeFile = koraliDir + '/variable.cpp'
 print('[Korali] Creating: ' + variableNewCodeFile + '...')
 with open(variableNewCodeFile, 'w') as file: file.write(variableCodeString)
 
 
 
 
