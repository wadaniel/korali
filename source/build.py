#!/usr/bin/env python3
import os
import json

#####################################################################

def getVariableType(v):
 if ('Real' in v["Type"]): cTypeString = 'double'
 if ('Signed Integer' in v["Type"]): cTypeString = 'int'
 if ('Unsigned Integer' in v["Type"]): cTypeString = 'size_t'
 if ('String' in v["Type"]): cTypeString = 'std::string'
 if ('Boolean' in v["Type"]): cTypeString = 'bool'
 
 arrayDepth = v["Type"].count("(Array of)")
 for x in range(arrayDepth):
  cTypeString = 'std::vector<' + cTypeString
  cTypeString = cTypeString + '>'
 
 return cTypeString
 
def getVariableName(v):
 cVarName = v["Name"].replace(" ", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName
 
def getVariableDefault(v):
 return v.get('Default', '')

#####################################################################

def consumeValue(base, moduleName, settingName, varName, varType, varDefault, path = []):
 cString = '\n'
 
 cString += ('  if (isDefined(' + base + ', {')
 for p in path: cString += (' "' + p +'",')
 cString += (' "' + settingName + '"})) \n  { \n')

 cString += ('   ' + varName + ' = ' + base )
 for p in path: cString += ('.at("' + p +'")')
 getLine = '.at("' + settingName + '").get<' + varType + '>();\n'
 cString += getLine
 
 cString += ('   ' + base)
 for p in path: cString += ('.at("' + p +'")')
 cString += ('.erase("' + settingName + '");\n')
 
 cString += ('  }\n')
 
 if (varDefault == 'Korali Skip Default'):
  return cString
 
 cString += ('  else ')
 if (varDefault == ''):
  cString += ('  koraliError("No value provided for mandatory setting: ')
  for p in path: cString += ('[' + p + '] > ')
  cString += ('[' + settingName + '], required by ' + moduleName + '.\\n"); ')
 else:
  if ("std::string" in varType): varDefault = '"' + varDefault + '"'
  defaultLine = varName + ' = ' + varDefault + ';'
  if ('std::vector<' in varType):
   defaultLine = 'for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i] = ' + varDefault + ';'
  if ('std::vector<std::vector<' in varType):
   defaultLine = 'for(size_t i = 0; i < ' + varName + '.size(); i++) ' + 'for(size_t j = 0; j < ' + varName + '[i].size(); j++) ' + varName + '[i][j] = ' + varDefault + ';'
  cString += (defaultLine)
   
 cString += ('\n')
 return cString

#####################################################################

# Obtaining list of solvers, problems, and conduits
solverPaths  = [x[0] for x in os.walk('./solvers')][1:]
problemPaths = [x[0] for x in os.walk('./problems')][1:]
conduitPaths = [x[0] for x in os.walk('./conduits')][1:]

# Creating new config.cpp
configFile = open('./config.cpp', 'w')
configFile.write('#include "korali.hpp"\n')

# Loading template variable header file
with open('./.variable.hpp', 'r') as file: variableHeaderString = file.read()
variableSettingDeclarationsString = '' 
variableSettingSet = set()

# Creating External Setting Strings for Variables
variableSetSolverSettingString = ''
variableGetSolverSettingString = ''
 
# Processing Solvers
for solverPath in solverPaths:
 solverName = solverPath.replace('./solvers/', '')
 
 # Loading JSON Configuration
 solverJsonFile = solverPath + '/' + solverName + '.json'
 with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
 solverConfig = json.loads(solverJsonString)
 
 ####### Producing solver.hpp
 
 # Producing private variable declarations
 solverSettingString = ''
 
 for v in solverConfig["Module Configuration"]:
  solverSettingString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
    
 for v in solverConfig["Termination Criteria"]:
  solverSettingString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
  solverSettingString += 'bool ' + getVariableName(v) + 'Enabled;\n'
  solverSettingString += 'bool ' + getVariableName(v) + 'Triggered;\n'
  
 for v in solverConfig["Internal Settings"]:
  solverSettingString += getVariableType(v) + ' ' + getVariableName(v) + ';\n'
      
 # Loading template header .hpp file
 solverTemplateHeaderFile = solverPath + '/.' + solverName + '.hpp'
 with open(solverTemplateHeaderFile, 'r') as file: solverHeaderString = file.read()
 newHeaderString = solverHeaderString.replace('private:', 'private: \n' + solverSettingString + '\n')
 
 # Saving new header .hpp file
 solverNewHeaderFile = solverPath + '/' + solverName + '.hpp'
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
   configFile.write(consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Internals' ]))
 
 for v in solverConfig["Termination Criteria"]:
   configFile.write('\n ' + getVariableName(v) + 'Enabled = false;')
   terminationString = consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Termination Criteria' ]) 
   terminationString = terminationString.replace(getVariableName(v) + ' = ', getVariableName(v) + 'Enabled = true;\n  ' + getVariableName(v) + ' = ')
   configFile.write(terminationString)
   configFile.write(consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"] + ' Triggered', getVariableName(v) + 'Triggered', 'bool', 'false', [ 'Solver', 'Termination Criteria' ]))
 
 variableSetSolverSettingString += ' if (_k->_js["Solver"] == "' + solverConfig["Module Alias"] + '")\n {\n'
 for v in solverConfig["Variables Configuration"]: 
   variableSetSolverSettingString += consumeValue('js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [  ])
 variableSetSolverSettingString += ' }\n'
 
 configFile.write('} \n\n')
 
 ###### Creating Solver Get Configuration routine
 
 configFile.write('void Korali::Solver::' + solverConfig["Module Alias"]  + '::getConfiguration() \n{\n\n')

 configFile.write(' _k->_js["Solver"]["Type"] = "' + solverConfig["Module Alias"] + '";\n')

 for v in solverConfig["Module Configuration"]: 
   configFile.write(' _k->_js["Solver"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n')
   
 for v in solverConfig["Internal Settings"]: 
   configFile.write(' _k->_js["Solver"]["Internals"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n')
 
 for v in solverConfig["Termination Criteria"]: 
   configFile.write(' if (' + getVariableName(v) + 'Enabled == true) _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n')
   configFile.write(' _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + ' Triggered"] = ' + getVariableName(v) + 'Triggered;\n')
 
 variableGetSolverSettingString += ' if (_k->_js["Solver"] == "' + solverConfig["Module Alias"] + '")\n {\n'
 for v in solverConfig["Variables Configuration"]: 
   variableGetSolverSettingString += '  js["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
 variableGetSolverSettingString += ' }\n\n'
 
 configFile.write(' } \n\n')
 
 ###### Finished Parsing Solver

# Saving variable solver configuration
configFile.write('void Korali::Variable::setSolverSettings(nlohmann::json& js)\n{\n')
configFile.write(variableSetSolverSettingString)
configFile.write('}\n\n')

# Saving variable solver configuration
configFile.write('void Korali::Variable::getSolverSettings(nlohmann::json& js)\n{\n')
configFile.write(variableGetSolverSettingString)
configFile.write('}\n\n')

# Saving new variable.hpp file
variableNewHeaderFile = './variable.hpp'
newHeaderString = variableHeaderString.replace('public:', 'public: \n' + variableSettingDeclarationsString + '\n')
with open(variableNewHeaderFile, 'w') as file: file.write(newHeaderString)

configFile.close()