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
 cVarType = v["Type"]
 cVarType = cVarType.replace('size_t', 'Unsigned Integer')
 cVarType = cVarType.replace('int', 'Integer')
 cVarType = cVarType.replace('double', 'Real Number')
 cVarType = cVarType.replace('std::string', 'String')
 cVarType = cVarType.replace('>','')
 cVarType = cVarType.replace('std::function<void(korali::Sample&)', 'Function(:ref:`Korali::Sample& <korali-sample>`)')
 cVarType = cVarType.replace('std::vector<std::vector<', 'List of Lists of ')
 cVarType = cVarType.replace('std::vector<', 'List of ')
 cVarType = cVarType.replace('bool', 'True/False')
 if ('korali::' in cVarType):
   print(cVarType.split())
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

def createVariableDescription(relPath, v): 
 moduleTypePath = ''
 if ("problem" in relPath): moduleTypePath = '["Problem"]'
 if ("solver" in relPath): moduleTypePath = '["Solver"]'
 if ("conduit" in relPath): moduleTypePath = '["Conduit"]'
 
 desc = '\n'
 desc += getJsonPath(v["Name"]).replace('"','').replace('[','').replace(']','') + '\n'
 desc += ' - **Usage**: e' + moduleTypePath + getJsonPath(v["Name"]).replace('\\','') + ' = ' + getDataType(v) + '\n'
 desc += ' - **Description**: ' + v["Description"] + '\n'
 if ('Criteria' in v): desc += ' - **Criteria**: :code:`' + v["Criteria"] + '`\n'
 if ('Options' in v):
   desc += ' - **Options**: \n\n'  
   for o in v["Options"]:
    desc += '      - "*' + o["Value"] + '*": ' + o["Description"] + '\n'
 desc += '\n'
 return desc
    
################################################
# Process Module Function

def processModule(parentModuleConfig, moduleRelPath, moduleName):
 modulePath = os.path.join(moduleSrcDir, moduleRelPath)
 moduleReadmeFile = modulePath + '/README.rst'
 moduleConfigFile = modulePath + '/' + moduleName + '.config'
 moduleOutputDir = '../modules/' + moduleRelPath

 moduleReadmeString = '.. _module-' + moduleName + ':\n\n'
  
 # Loading configuration and readme files 
 with open(moduleConfigFile, 'r') as file: moduleConfigString  = file.read()
 with open(moduleReadmeFile, 'r') as file: moduleReadmeString += file.read()

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

  if ('Compatible Solvers' in moduleConfig):
   moduleReadmeString += '\n**Compatible Solvers**\n'
   moduleReadmeString += '----------------------------------\n\n'
   moduleReadmeString += 'This problem can be solved using the following modules: \n\n'
   for v in moduleConfig["Compatible Solvers"]:
    moduleReadmeString += '   - :ref:`' + v + ' <module-' + v + '>`\n'

  if ('Variables Configuration' in moduleConfig):
   moduleReadmeString += '\n**Variable-Specific Settings**\n'
   moduleReadmeString += '----------------------------------\n\n'
   moduleReadmeString += 'These are settings required by this module that are added to each of the experiment\'s variables when this module is selected.\n\n'
   for v in moduleConfig["Variables Configuration"]:
    moduleReadmeString += createVariableDescription(moduleRelPath, v)
            
  moduleReadmeString += '\n**Configuration**\n'
  moduleReadmeString += '-----------------------------\n'
  moduleReadmeString += 'These are settings required by this module.\n\n'
  if ('Configuration Settings' in moduleConfig):
   for v in moduleConfig["Configuration Settings"]:
    moduleReadmeString += createVariableDescription(moduleRelPath, v)
  else:
    moduleReadmeString += '\n*None*\n'  

  if ('Termination Criteria' in moduleConfig):
   moduleReadmeString += '\n**Termination Criteria**\n'
   moduleReadmeString += '----------------------------------\n\n'
   moduleReadmeString += 'These are the customizable criteria that indicates whether the solver should continue or finish execution. Korali will stop when at least one of these conditions are met. The criteria is expressed in C++ since it is compiled and evaluated as seen here in the engine. \n\n'
   for v in moduleConfig["Termination Criteria"]:
    moduleReadmeString += createVariableDescription(moduleRelPath, v)

  if ('Internal Settings' in moduleConfig):
   moduleReadmeString += '\n**Internal Settings** *[For Developers]*\n'
   moduleReadmeString += '--------------------------------------------------\n\n'
   moduleReadmeString += 'The following are settings that store the internal state of the module. The information below is only interesting for developers and the user does not need to set them up. \n\n'
   for v in moduleConfig["Internal Settings"]:
    moduleReadmeString += createVariableDescription(moduleRelPath, v)
    
  if ('Module Defaults' in moduleConfig):
   moduleReadmeString += '\n**Default Configuration**\n'
   moduleReadmeString += '----------------------------------\n\n'
   moduleReadmeString += 'These following configuration will be assigned by default. Any settings defined by the user will override the given settings specified in these defaults.\n\n'
   moduleReadmeString += '  .. code-block:: python\n\n'
   moduleReadmeString += '    ' + json.dumps(moduleConfig['Module Defaults'], sort_keys=True, indent=4).replace('}','    }')
   
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
