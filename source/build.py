#!/usr/bin/env python3
import os
import json

def getVariableDeclaration(v):
 if ('Real' in v["Type"]): cTypeString = 'double'
 if ('Signed Integer' in v["Type"]): cTypeString = 'int'
 if ('Unsigned Integer' in v["Type"]): cTypeString = 'size_t'
 if ('String' in v["Type"]): cTypeString = 'std::string'
 if ('Boolean' in v["Type"]): cTypeString = 'bool'
 
 arrayDepth = v["Type"].count("(Array of)")
 for x in range(arrayDepth):
  cTypeString = 'std::vector<' + cTypeString
  cTypeString = cTypeString + '>'
   
 cVarName = v["Name"].replace(" ", "")
 cVarName = cVarName[0].lower() + cVarName[1:]
 cVarDeclaration = cTypeString + ' _' + cVarName
 return cVarDeclaration

# Obtaining list of solvers, problems, and conduits
solverPaths  = [x[0] for x in os.walk('./solvers')][1:]
problemPaths = [x[0] for x in os.walk('./problems')][1:]
conduitPaths = [x[0] for x in os.walk('./conduits')][1:]

# Loading template variable header file
variableTemplateHeaderFile = './.variable.hpp'
with open(variableTemplateHeaderFile, 'r') as file: variableHeaderString = file.read()
variableSettingString = '' 
variableSettingSet = set()
 
# Processing Solvers
for solverPath in solverPaths:
 solverName = solverPath.replace('./solvers/', '')
 
 # Loading JSON Configuration
 solverJsonFile = solverPath + '/' + solverName + '.json'
 with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
 solverConfig = json.loads(solverJsonString)
 
 # Producing private variable declarations
 solverSettingString = ''
 
 for v in solverConfig["Module Configuration"]:
  solverSettingString += getVariableDeclaration(v) + ';\n'
    
 for v in solverConfig["Termination Criteria"]:
  solverSettingString += getVariableDeclaration(v) + ';\n'
  solverSettingString += getVariableDeclaration(v) + '_enabled;\n'
  solverSettingString += getVariableDeclaration(v) + '_triggered;\n'
  
 for v in solverConfig["Internal Settings"]:
  solverSettingString += getVariableDeclaration(v) + ';\n'
      
 # Loading template header .hpp file
 solverTemplateHeaderFile = solverPath + '/.' + solverName + '.hpp'
 with open(solverTemplateHeaderFile, 'r') as file: solverHeaderString = file.read()
 newHeaderString = solverHeaderString.replace('private:', 'private: \n' + solverSettingString + '\n')
 
 # Saving new header .hpp file
 solverNewHeaderFile = solverPath + '/' + solverName + '.hpp'
 with open(solverNewHeaderFile, 'w') as file: file.write(newHeaderString)
 
 # Reading variable-specific configuration
 for v in solverConfig["Variables Configuration"]:
  if (not v["Name"] in variableSettingSet):
   variableSettingString += getVariableDeclaration(v) + ';\n'
   variableSettingSet.add(v["Name"])

# Saving new variable.hpp file
variableNewHeaderFile = './variable.hpp'
newHeaderString = variableHeaderString.replace('public:', 'public: \n' + variableSettingString + '\n')
with open(variableNewHeaderFile, 'w') as file: file.write(newHeaderString)
  