#!/usr/bin/env python3

import sys
import os
import json
import shutil

moduleSrcDir = '../../../source/modules'

################################################
# Auxiliar Functions
def getVariableName(v):
 cVarName = v["Name"].replace(" ", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName

def getDataType(v):
 cVarType = v["Type"].replace('size_t', 'Unsigned Integer').replace('int', 'Integer').replace('double', 'Real Number').replace('std::string', 'String').replace('std::vector<std::vector<', 'List of Lists of ').replace('std::vector<', 'List of ').replace('>','').replace('bool', 'True/False')
 return cVarType 
 
def getJsonPath(path):
 nameString = ''
 for item in path:
  nameString += '\\["' + item + '"\\]'
 return nameString
 
################################################
# Process Module Function

def processModule(moduleRelPath, moduleName):
 modulePath = os.path.join(moduleSrcDir, moduleRelPath)
 moduleReadmeFile = modulePath + '/README.rst'
 moduleConfigFile = modulePath + '/' + moduleName + '.config'
 moduleOutputDir = '../modules/' + moduleRelPath
 
 # Loading configuration and readme files 
 with open(moduleConfigFile, 'r') as file: moduleConfigString = file.read()
 with open(moduleReadmeFile, 'r') as file: moduleReadmeString = file.read()

 # Loading Module's Configuration
 moduleConfig = json.loads(moduleConfigString)
 
 # Creating module's folder, if not exists
 if not os.path.exists(moduleOutputDir):
  os.mkdir(moduleOutputDir) 
  
 # Creating subfolder list
 subFolderList = []
 list_dir = os.listdir(fullPath)
 for f in list_dir:
  if not os.path.isfile(os.path.join(fullPath, f)):
   subFolderList.append(f)
   
 # Determining if its a parent or leaf module
 isParentModule = True
 if (subFolderList == []): isParentModule = False
  
 # If its parent, construct children modules
 if (isParentModule == True):
  moduleReadmeString += '**Sub-Categories**\n\n'
  moduleReadmeString += '  .. toctree::\n'
  moduleReadmeString += '     :maxdepth: 1\n'
  
  for f in subFolderList:
   subModuleFullPath = os.path.join(modulePath, f)
   subModuleConfigFile = subModuleFullPath + '/' + f + '.config'
   with open(subModuleConfigFile, 'r') as file: subModuleConfigString = file.read()
   subModuleConfig = json.loads(subModuleConfigString)
   moduleReadmeString += '    ' + f + '/' + f + '\n'
 
   subModuleDstPath = moduleOutputDir + '/' + f
   if not os.path.exists(subModuleDstPath):
    os.mkdir(subModuleDstPath) 
     
 # If its leaf, build configuration
 if (isParentModule == False): 
  moduleReadmeString += '\n**Configuration**\n'
 
  if ('Configuration Settings' in moduleConfig):
   for v in moduleConfig["Configuration Settings"]:
    moduleReadmeString += '\n'
    moduleReadmeString += getJsonPath(v["Name"]) + '\n'
    moduleReadmeString += ' - **Type**: ' + getDataType(v) + '\n'
    moduleReadmeString += ' - **Description**: ' + v["Description"] + '\n'
    moduleReadmeString +='\n'  

 # Saving Module's readme file
 moduleReadmeString += '\n\n'
 with open(moduleOutputDir + '/' + moduleName + '.rst', 'w') as file: file.write(moduleReadmeString)

############################################
# Main Procedure

shutil.rmtree('../modules', ignore_errors=True, onerror=None)
os.makedirs('../modules')

for root, dirs, files in os.walk(moduleSrcDir, topdown=True):
 for name in dirs:
  fullPath = os.path.join(root, name)
  if (not '.o/' in fullPath and not '.d/' in fullPath):
   relPath = os.path.relpath(fullPath, moduleSrcDir)
   processModule(relPath, name)
     
     