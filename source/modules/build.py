import sys
import os
import json
from buildAux import *

def buildModules(baseDir):
 # modules List
 moduleDetectionList = ''
 
 # Detecting modules' json file
 for moduleDir, d, fileNames in os.walk(baseDir):
  for fileName in fileNames: 
   if '.json' in fileName:
    with open(moduleDir + '/' + fileName, 'r') as file: moduleConfig = json.load(file)
    moduleName = fileName.replace('.json', '')

    ####### Adding module to list
  
    moduleDetectionList += '  if(moduleType == "' + moduleConfig["Alias"] + '") module = new ' + moduleConfig["C++ Class"] + '();\n'
    
    ###### Producing module code
    
    ## Adding getType()
    getTypeString = 'std::string ' + moduleConfig["C++ Class"] + '::getType() { return "' + moduleConfig["Alias"] + '"; }'
    
    moduleCodeString = createSetConfiguration(moduleConfig)
    moduleCodeString += createGetConfiguration(moduleConfig)
    moduleCodeString += createCheckTermination(moduleConfig)
    moduleCodeString += '\n\n' + getTypeString + '\n\n'
  
    ####### Producing header file
    
    # Loading template header .hpp file
    moduleTemplateHeaderFile = moduleDir + '/' + moduleName + '._hpp'
    with open(moduleTemplateHeaderFile, 'r') as file: moduleTemplateHeaderString = file.read()
    
    # Adding declarations
    declarationsString = createHeaderDeclarations(moduleConfig)
    newHeaderString = moduleTemplateHeaderString.replace('public:', 'public: \n' + declarationsString + '\n')
    
    # Saving new header .hpp file
    moduleNewHeaderFile = moduleDir + '/' + moduleName + '.hpp'
    print('[Korali] Creating: ' + moduleNewHeaderFile + '...')
    with open(moduleNewHeaderFile, 'w') as file: file.write(newHeaderString)
    
    ###### Creating code file
    
    moduleBaseCodeFileName = moduleDir + '/' + moduleName + '._cpp'
    moduleNewCodeFile = moduleDir + '/' + moduleName + '.cpp'
    baseFileTime = os.path.getmtime(moduleBaseCodeFileName)
    newFileTime = baseFileTime
    if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)
    
    if (baseFileTime >= newFileTime):
     with open(moduleBaseCodeFileName, 'r') as file: moduleBaseCodeString = file.read()
     moduleBaseCodeString += '\n\n' + moduleCodeString
     print('[Korali] Creating: ' + moduleNewCodeFile + '...')
     with open(moduleNewCodeFile, 'w') as file: file.write(moduleBaseCodeString)

 ###### Updating module configuration file 

 moduleBaseCodeFileName = baseDir + '/module._cpp' 
 moduleNewCodeFile = baseDir + '/module.cpp'
 baseFileTime = os.path.getmtime(moduleBaseCodeFileName)
 newFileTime = baseFileTime
 if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)

 if (baseFileTime >= newFileTime):
   with open(moduleBaseCodeFileName, 'r') as file: moduleBaseCodeString = file.read()
   newBaseString = moduleBaseCodeString.replace(' // Module List', ' // Module List' + moduleDetectionList)
   print('[Korali] Creating: ' + moduleNewCodeFile + '...')
   with open(moduleNewCodeFile, 'w') as file: file.write(newBaseString)

 ###### Creating base header file
 
 moduleBaseHeaderFileName = baseDir + '/module._hpp'
 moduleNewHeaderFile = baseDir + '/module.hpp'
 with open(moduleBaseHeaderFileName, 'r') as file: moduleBaseHeaderString = file.read()
 newBaseString = moduleBaseHeaderString
 print('[Korali] Creating: ' + moduleNewHeaderFile + '...')
 with open(moduleNewHeaderFile, 'w+') as file: file.write(newBaseString)