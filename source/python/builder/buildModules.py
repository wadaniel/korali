import sys
import os
import json
from buildAux import *

def buildModules(moduleDir):
 # modules List
 moduleDetectionList = ''
 
 # Variable Declarations List
 moduleDeclarationsSet = set()
 
 # Detecting modules
 modulePaths  = [x[0] for x in os.walk(moduleDir)][1:]
 for modulePath in modulePaths:
  moduleName = modulePath.replace(moduleDir + '/', '')
  
  # Loading JSON Configuration
  moduleJsonFile = modulePath + '/' + moduleName + '.json'
  if (not os.path.isfile(moduleJsonFile)): continue 
  with open(moduleJsonFile, 'r') as file: moduleJsonString = file.read()
  moduleConfig = json.loads(moduleJsonString)
  
  ####### Adding module to list

  moduleDetectionList += '  if(moduleType == "' + moduleConfig["Alias"] + '") module = new ' + moduleConfig["C++ Class"] + '();\n'
  
  ###### Producing module code

  for declaration in createHeaderDeclarations(moduleConfig).splitlines():
   moduleDeclarationsSet.add(declaration)
  
  ## Adding getType()
  getTypeString = 'std::string ' + moduleConfig["C++ Class"] + '::getType() { return "' + moduleConfig["Alias"] + '"; }'
  
  moduleCodeString = createSetConfiguration(moduleConfig)
  moduleCodeString += createGetConfiguration(moduleConfig)
  moduleCodeString += createCheckTermination(moduleConfig)
  moduleCodeString += '\n\n' + getTypeString + '\n\n'

  ####### Producing header file
  
  # Loading template header .hpp file
  moduleTemplateHeaderFile = modulePath + '/' + moduleName + '._hpp'
  with open(moduleTemplateHeaderFile, 'r') as file: moduleTemplateHeaderString = file.read()
  newHeaderString = moduleTemplateHeaderString
  
  # Saving new header .hpp file
  moduleNewHeaderFile = modulePath + '/' + moduleName + '.hpp'
  print('[Korali] Creating: ' + moduleNewHeaderFile + '...')
  with open(moduleNewHeaderFile, 'w') as file: file.write(newHeaderString)
  
  ###### Creating code file
  
  moduleBaseFileName = modulePath + '/' + moduleName + '._cpp'
  moduleNewCodeFile = modulePath + '/' + moduleName + '.cpp'
  baseFileTime = os.path.getmtime(moduleBaseFileName)
  newFileTime = baseFileTime
  if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)
  
  if (baseFileTime >= newFileTime):
   with open(moduleBaseFileName, 'r') as file: moduleBaseCodeString = file.read()
   moduleBaseCodeString += '\n\n' + moduleCodeString
   print('[Korali] Creating: ' + moduleNewCodeFile + '...')
   with open(moduleNewCodeFile, 'w') as file: file.write(moduleBaseCodeString)

 ###### Creating base configuration file

 moduleBaseFileName = moduleDir + '/base._cpp'
 moduleNewCodeFile = moduleDir + '/base.cpp'
 baseFileTime = os.path.getmtime(moduleBaseFileName)
 newFileTime = baseFileTime
 if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)

 if (baseFileTime >= newFileTime):
   with open(moduleBaseFileName, 'r') as file: moduleBaseCodeString = file.read()
   newBaseString = moduleBaseCodeString.replace(' // Module List', moduleDetectionList)
   with open(moduleNewCodeFile, 'w') as file: file.write(newBaseString)

 ###### Creating base header file
 declarationsString = ''
 for declaration in moduleDeclarationsSet:
  declarationsString += declaration + '\n'
 
 with open(moduleDir + '/base._hpp', 'r') as file: moduleBaseHeaderString = file.read()
 newBaseString = moduleBaseHeaderString.replace('public:', 'public: \n' + declarationsString + '\n')
 with open(moduleDir + '/base.hpp', 'w+') as file: file.write(newBaseString)