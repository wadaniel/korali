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
 cVarType = v["Type"].replace('size_t', 'Unsigned Integer').replace('int', 'Integer').replace('double', 'Real Number').replace('std::string', 'String').replace('std::vector<std::vector<', 'List of Lists of ').replace('std::vector<', 'List of ').replace('>','').replace('bool', 'True/False')
 return cVarType 
 
def getJsonPath(path):
 nameString = ''
 for item in path:
  nameString += '\\["' + item + '"\\]'
 return nameString
 
def processLeafModuleSubFolder(modulePath):
 moduleDir = '../../../source/' + modulePath
 moduleName = os.path.basename(os.path.normpath(modulePath))
 
 moduleConfigFile = moduleDir + '/' + moduleName + '.config'
 moduleReadmeFile = moduleDir + '/README.rst'
 
 with open(moduleReadmeFile, 'r') as file: moduleReadmeString = file.read()
 with open(moduleConfigFile, 'r') as file: moduleConfigString = file.read()

 moduleConfig = json.loads(moduleConfigString)
 
 moduleReadmeString += '\nConfiguration\n'
 moduleReadmeString += '#####################\n\n'
 
 if ('Configuration Settings' in moduleConfig):
  for v in moduleConfig["Configuration Settings"]:
   moduleReadmeString += '\n'
   moduleReadmeString += getJsonPath(v["Name"]) + '\n'
   moduleReadmeString += ' - **Type**: ' + getDataType(v) + '\n'
   moduleReadmeString += ' - **Description**: ' + v["Description"] + '\n'
   moduleReadmeString +='\n'  
    
 moduleOutputDir = '../modules/'
 with open(moduleOutputDir + '/' + moduleName + '.rst', 'w') as file: file.write(moduleReadmeString)
 
def processParentModuleSubFolder(modulePath):
 moduleDir = '../../../source/' + modulePath
 moduleName = os.path.basename(os.path.normpath(modulePath))
 
 moduleReadmeFile = moduleDir + '/README.rst'
 with open(moduleReadmeFile, 'r') as file: moduleReadmeString = file.read()

 #moduleReadmeString += '  .. toctree::\n'
 #moduleReadmeString += '     :maxdepth: 1\n'
 #moduleReadmeString += '     :caption: Sub-Categories\n'

 moduleOutputDir = '../modules/' + modulePath
 if not os.path.exists(moduleOutputDir):
  os.mkdir(moduleOutputDir) 
 with open(moduleOutputDir + '/' + moduleName + '.rst', 'w') as file: file.write(moduleReadmeString)
     
 list_dir = os.listdir(moduleDir)
 for f in list_dir:
  subModuleFullPath = os.path.join(moduleDir, f)
  if not os.path.isfile(subModuleFullPath):
   subModuleConfigFile = subModuleFullPath + '/' + f + '.config'
   with open(subModuleConfigFile, 'r') as file: subModuleConfigString = file.read()
   subModuleConfig = json.loads(subModuleConfigString)
   moduleReadmeString += ' ' + f + ' \n'

   subModuleDstPath = moduleOutputDir + '/' + f
   if not os.path.exists(subModuleDstPath):
     os.mkdir(subModuleDstPath) 
  
shutil.rmtree('../modules', ignore_errors=True, onerror=None)
os.makedirs('../modules')

processLeafModuleSubFolder('engine')
processLeafModuleSubFolder('experiment')
processParentModuleSubFolder('solver')
processParentModuleSubFolder('problem')
processParentModuleSubFolder('conduit')

for root, dirs, files in os.walk("../../../source/solver", topdown=True):
 for name in dirs:
  fullPath = os.path.join(root, name)
  if (not '.o/' in fullPath and not '.d/' in fullPath):
   list_dir = os.listdir(fullPath)
   hasSubFolders = False
   for f in list_dir:
    if not os.path.isfile(os.path.join(fullPath, f)):
     hasSubFolders = True
   print(fullPath)
   #if (hasSubFolders):
   #  print('Has Subfolders')
   #else:
   #  print('Has No Subfolders')