#!/usr/bin/env python3

import sys
import os
import json
import shutil
from buildAux import *

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
