#!/usr/bin/env python3

import sys
import os
import json
import shutil
import copy

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

def recursiveUpdate(dest, defaults):
 if (isinstance(defaults, dict)):
  for k, x in defaults.items():
   if (not k in dest): dest[k] = copy.deepcopy(defaults[k])
   else: 
     recursiveUpdate(dest[k], defaults[k])
 if (isinstance(defaults, list)):
  for k in range(len(defaults)):
   if (not k in dest): dest.append(copy.deepcopy(defaults[k]))
   else: 
     recursiveUpdate(dest[k], defaults[k])
 
################################################
# Process Module Function

def processModule(parentModuleConfig, moduleRelPath, moduleName):
 modulePath = os.path.join(moduleSrcDir, moduleRelPath)
 moduleReadmeFile = modulePath + '/README.rst'
 moduleConfigFile = modulePath + '/' + moduleName + '.config'
 moduleOutputDir = '../modules/' + moduleRelPath
 
 # Loading configuration and readme files 
 with open(moduleConfigFile, 'r') as file: moduleConfigString = file.read()
 with open(moduleReadmeFile, 'r') as file: moduleReadmeString = file.read()

 # Loading Module's Configuration
 moduleConfig = json.loads(moduleConfigString)
 recursiveUpdate(moduleConfig, parentModuleConfig)
 
 # Creating module's folder, if not exists
 if not os.path.exists(moduleOutputDir):
  os.mkdir(moduleOutputDir) 
  
 # Creating subfolder list
 subFolderList = []
 list_dir = os.listdir(modulePath)
 for f in list_dir:
  fullPath = os.path.join(modulePath, f)
  if not os.path.isfile(fullPath):
   if (not '.o/' in fullPath and not '.d/' in fullPath):
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
   moduleReadmeString += '    ' + f + '/' + f + '\n'
   subPath = os.path.join(moduleRelPath, f)
   processModule(moduleConfig, subPath, f)
   
 # If its leaf, build configuration
 if (isParentModule == False): 
  if ('Termination Criteria' in moduleConfig):
   moduleReadmeString += '\n**Termination Criteria**\n'
   for v in moduleConfig["Termination Criteria"]:
    moduleReadmeString += '\n'
    moduleReadmeString += getJsonPath(v["Name"]) + '\n'
    moduleReadmeString += ' - **Type**: ' + getDataType(v) + '\n'
    moduleReadmeString += ' - **Criterion**: ' + v["Criteria"] + '\n'
    moduleReadmeString += ' - **Description**: ' + v["Description"] + '\n'
    moduleReadmeString +='\n'
    
  moduleReadmeString += '\n**Configuration**\n'
  if ('Configuration Settings' in moduleConfig):
   for v in moduleConfig["Configuration Settings"]:
    moduleReadmeString += '\n'
    moduleReadmeString += getJsonPath(v["Name"]) + '\n'
    moduleReadmeString += ' - **Type**: ' + getDataType(v) + '\n'
    moduleReadmeString += ' - **Description**: ' + v["Description"] + '\n'
    moduleReadmeString +='\n'
  else:
    moduleReadmeString += '\n*None*\n'  

 # Saving Module's readme file
 moduleReadmeString += '\n\n'
 with open(moduleOutputDir + '/' + moduleName + '.rst', 'w') as file: file.write(moduleReadmeString)

############################################
# Main Procedure

shutil.rmtree('../modules', ignore_errors=True, onerror=None)
os.makedirs('../modules')

list_dir = os.listdir(moduleSrcDir)
for f in list_dir:
 fullPath = os.path.join(moduleSrcDir, f)
 if not os.path.isfile(fullPath):
  if (not '.o/' in fullPath and not '.d/' in fullPath):
   processModule({}, f, f)
