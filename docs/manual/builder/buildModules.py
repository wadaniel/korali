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

def createVariableDescription(v):
 desc = '\n'
 desc += getJsonPath(v["Name"]) + '\n'
 desc += ' - **Type**: ' + getDataType(v) + '\n'
 desc += ' - **Description**: ' + v["Description"] + '\n'
 if ('Criteria' in v): desc += ' - **Criteria**: ' + v["Criteria"] + '\n'
 desc += '\n'
 return desc
    
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
  moduleReadmeString += '.. toctree::\n'
  moduleReadmeString += '   :titlesonly:\n\n'
  
  for f in subFolderList:
   subModuleFullPath = os.path.join(modulePath, f)
   moduleReadmeString += '   ' + f + '/' + f + '\n'
   subPath = os.path.join(moduleRelPath, f)
   processModule(moduleConfig, subPath, f)
   
 # If its leaf, build configuration
 if (isParentModule == False): 
    
  moduleReadmeString += '\n**Configuration**\n'
  moduleReadmeString += '-----------------------------\n'
  if ('Configuration Settings' in moduleConfig):
   for v in moduleConfig["Configuration Settings"]:
    moduleReadmeString += createVariableDescription(v)
  else:
    moduleReadmeString += '\n*None*\n'  

  if ('Termination Criteria' in moduleConfig):
   moduleReadmeString += '\n**Termination Criteria**\n'
   moduleReadmeString += '-----------------------------\n'
   for v in moduleConfig["Termination Criteria"]:
    moduleReadmeString += createVariableDescription(v)

  if ('Internal Settings' in moduleConfig):
   moduleReadmeString += '\n*[For Developers]* **Internal Settings**\n'
   moduleReadmeString += '----------------------------------------------------\n'
   for v in moduleConfig["Internal Settings"]:
    moduleReadmeString += createVariableDescription(v)
    
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
