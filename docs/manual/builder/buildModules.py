#!/usr/bin/env python3

import sys
import os
import json
import shutil

def getVariableName(v):
 cVarName = v["Name"].replace(" ", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName

def getDataType(v):
 cVarType = v["Type"].replace('size_t', 'Unsigned Integer').replace('int', 'Integer').replace('double', 'Real Number').replace('std::string', 'String').replace('std::vector<', 'Array of ').replace('<','')
 return cVarType 
 
def getJsonPath(path):
 nameString = ''
 for item in path:
  nameString += '["' + item + '"]'
 return nameString
 
def getVariableInfo(v, moduleName):
 varString = ''
 varString += '??? abstract "' + getJsonPath(v["Name"]) + '"\n\n'
 varString += '\t' + v["Description"] + '\n'
 varString += '\n'
 
 if ('Korali::Distribution' in v["Type"]):
  with open('distributionList.txt', 'r') as file: distributionListString = file.read()
  varString += '\t**Available Distributions**: \n'
  varString += distributionListString
  return varString
  
 varString += '\t+ Type: ' + getDataType(v) + '\n'
 varString += '\t+ Description: v["Description"]\n\n' 
 return varString
 
def processModuleSubFolder(modulePath):
 moduleDir = '../../../source/' + modulePath
 moduleName = os.path.basename(os.path.normpath(modulePath))
 
 moduleConfigFile = moduleDir + '/' + moduleName + '.config'
 moduleReadmeFile = moduleDir + '/README.rst'
 
 with open(moduleReadmeFile, 'r') as file: moduleReadmeString = file.read()
 with open(moduleConfigFile, 'r') as file: moduleConfigString = file.read()

 moduleConfig = json.loads(moduleConfigString)
 
 moduleReadmeString += '\n\nModule Configuration\n'
 moduleReadmeString += '#####################\n\n'
 
 if ('Configuration Settings' in moduleConfig):
  for v in moduleConfig["Configuration Settings"]:
   moduleReadmeString += getVariableInfo(v, moduleName)
  moduleReadmeString +='\n\n'  
    
 moduleOutputDir = '../modules/'
 with open(moduleOutputDir + '/' + moduleName + '.rst', 'w') as file: file.write(moduleReadmeString)
 
shutil.rmtree('../modules', ignore_errors=True, onerror=None)
os.makedirs('../modules')

processModuleSubFolder('engine')
processModuleSubFolder('experiment')
processModuleSubFolder('solver')
processModuleSubFolder('problem')
processModuleSubFolder('conduit')
