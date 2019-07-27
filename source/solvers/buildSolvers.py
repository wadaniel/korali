import sys
import os
import json
sys.path.append('../')
from auxiliar import *

def buildSolvers():
 # Processing Solvers
 curdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 
 
 # Creating new config.cpp
 configFileName = curdir + '/config.cpp'
 configFile = open(configFileName, 'w')
 configFile.write('#include "korali.hpp"\n')

 # Creating variable settings
 variableSettingDeclarationsString = '' 
 variableSettingSet = set()
 
 # Creating External Setting Strings for Variables
 variableSetSolverSettingString = ''
 variableGetSolverSettingString = ''
 
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
  solverSettingString = ''
  
  for v in solverConfig["Module Configuration"]:
   solverSettingString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
     
  for v in solverConfig["Termination Criteria"]:
   solverSettingString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
   solverSettingString += 'int ' + getVariableName(v) + 'Enabled;\n'
   solverSettingString += 'int ' + getVariableName(v) + 'Triggered;\n'
   
  for v in solverConfig["Internal Settings"]:
   solverSettingString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
       
  # Loading template header .hpp file
  solverTemplateHeaderFile = solverPath + '/.' + solverName + '.hpp'
  with open(solverTemplateHeaderFile, 'r') as file: solverHeaderString = file.read()
  newHeaderString = solverHeaderString.replace('private:', 'private: \n' + solverSettingString + '\n')
  
  # Saving new header .hpp file
  solverNewHeaderFile = solverPath + '/' + solverName + '.hpp'
  print('[Korali] Creating: ' + solverNewHeaderFile + '...')
  with open(solverNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ####### Adding information to variable.hpp
  
  # Reading variable-specific configuration
  for v in solverConfig["Variables Configuration"]:
   if (not v["Name"] in variableSettingSet):
    variableSettingDeclarationsString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
    variableSettingSet.add(v["Name"])
    
  ###### Creating Solver Set Configuration routine
  
  configFile.write('void Korali::Solver::' + solverConfig["Module Alias"] + '::setConfiguration() \n{\n')
 
  # Consume Solver Settings
  for v in solverConfig["Module Configuration"]:
    configFile.write(consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [ 'Solver' ]))
  
  for v in solverConfig["Internal Settings"]:
    configFile.write(consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Internal' ]))
  
  for v in solverConfig["Termination Criteria"]:
    configFile.write('\n ' + getVariableName(v) + 'Enabled = 0;')
    terminationString = consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Termination Criteria' ]) 
    terminationString = terminationString.replace(getVariableName(v) + ' = ', getVariableName(v) + 'Enabled = 1;\n  ' + getVariableName(v) + ' = ')
    configFile.write(terminationString)
    configFile.write(consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"] + ' Triggered', getVariableName(v) + 'Triggered', 'int', '0', [ 'Solver', 'Termination Criteria' ]))
  
  configFile.write('} \n\n')
  
  # Creating set variable configuration
  
  variableSetSolverSettingString += ' if (isDefined(_k->_js, { "Solver", "Type" })) if (_k->_js["Solver"]["Type"] == "' + solverConfig["Module Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableSetSolverSettingString += consumeValue('js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [  ])
  variableSetSolverSettingString += ' }\n'
  
  ###### Creating Solver Get Configuration routine
  
  configFile.write('void Korali::Solver::' + solverConfig["Module Alias"]  + '::getConfiguration() \n{\n\n')
 
  configFile.write(' _k->_js["Solver"]["Type"] = "' + solverConfig["Module Alias"] + '";\n')
 
  for v in solverConfig["Module Configuration"]: 
    configFile.write(' _k->_js["Solver"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n')
    
  for v in solverConfig["Internal Settings"]: 
    configFile.write(' _k->_js["Solver"]["Internal"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n')
  
  for v in solverConfig["Termination Criteria"]: 
    configFile.write(' if (' + getVariableName(v) + 'Enabled == 1) _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n')
    configFile.write(' _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + ' Triggered"] = ' + getVariableName(v) + 'Triggered;\n')
  
  configFile.write(' } \n\n')
  
  # Creating get variable configuration
  
  variableGetSolverSettingString += ' if (isDefined(_k->_js, { "Solver", "Type" })) if (_k->_js["Solver"]["Type"] == "' + solverConfig["Module Alias"] + '")\n {\n'
  for v in solverConfig["Variables Configuration"]: 
    variableGetSolverSettingString += '  js["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
  variableGetSolverSettingString += ' }\n\n'
 
  
 # Saving variable solver configuration
 configFile.write('void Korali::Variable::setSolverSettings(nlohmann::json& js)\n{\n')
 configFile.write(variableSetSolverSettingString)
 configFile.write('}\n\n')
 
 # Saving variable solver configuration
 configFile.write('void Korali::Variable::getSolverSettings(nlohmann::json& js)\n{\n')
 configFile.write(variableGetSolverSettingString)
 configFile.write('}\n\n')
 
 ###### Finished Parsing Solver
  
 print('[Korali] Creating: ' + configFileName + '...')
 configFile.close()
 return variableSettingDeclarationsString
