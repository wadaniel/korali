import sys
import os
import json

#####################################################################

def getVariableType(v):
 # Replacing bools with ints for Python compatibility
 return v['Type'].replace('bool', 'int')
 
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
  
  for v in solverConfig["Module Configuration"]:
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
  
  solverCodeString = 'void Korali::Solver::' + solverConfig["Module Alias"] + '::setConfiguration() \n{\n'
 
  # Consume Solver Settings
  for v in solverConfig["Module Configuration"]:
    solverCodeString += consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), getVariableDefault(v), [ 'Solver' ])
  
  for v in solverConfig["Internal Settings"]:
    solverCodeString += consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Internal' ])
  
  for v in solverConfig["Termination Criteria"]:
    solverCodeString += '\n ' + getVariableName(v) + 'Enabled = 0;'
    terminationString = consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"], getVariableName(v), getVariableType(v), 'Korali Skip Default', [ 'Solver', 'Termination Criteria' ]) 
    solverCodeString += terminationString.replace(getVariableName(v) + ' = ', getVariableName(v) + 'Enabled = 1;\n  ' + getVariableName(v) + ' = ')
    solverCodeString += consumeValue('_k->_js', solverConfig["Module Alias"], v["Name"] + ' Triggered', getVariableName(v) + 'Triggered', 'int', '0', [ 'Solver', 'Termination Criteria' ])
  
  solverCodeString += '} \n\n'
  
  ###### Creating Solver Get Configuration routine
  
  solverCodeString += 'void Korali::Solver::' + solverConfig["Module Alias"]  + '::getConfiguration() \n{\n\n'
  solverCodeString += ' _k->_js["Solver"]["Type"] = "' + solverConfig["Module Alias"] + '";\n'
 
  for v in solverConfig["Module Configuration"]: 
    solverCodeString += ' _k->_js["Solver"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
    
  for v in solverConfig["Internal Settings"]: 
    solverCodeString += ' _k->_js["Solver"]["Internal"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
  
  for v in solverConfig["Termination Criteria"]: 
    solverCodeString += ' if (' + getVariableName(v) + 'Enabled == 1) _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + '"] = ' + getVariableName(v) + ';\n'
    solverCodeString += ' _k->_js["Solver"]["Termination Criteria"]["' + v["Name"] + ' Triggered"] = ' + getVariableName(v) + 'Triggered;\n'
  
  solverCodeString += ' } \n\n'
  
  ###### Creating code file
  with open(solverPath + '/' + solverName + '._cpp', 'r') as file: solverBaseCodeString = file.read()
  solverBaseCodeString += '\n\n' + solverCodeString
  solverNewCodeFile = solverPath + '/' + solverName + '.cpp'
  print('[Korali] Creating: ' + solverNewCodeFile + '...')
  with open(solverNewCodeFile, 'w') as file: file.write(solverBaseCodeString)
