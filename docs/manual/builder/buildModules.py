#!/usr/bin/env python3

import sys
import os
import json
from buildAux import *

def buildModules(koraliDir):
 # Processing Modules
 
 inputDirs = []
 inputFiles = []
 outputDirs = []
 outputFiles = []
 
 for root, dirs, files in os.walk(koraliDir, topdown=False):
   for name in files:
    if 'README.md' in name:
      inputDir = root
      inputFile = os.path.join(root, name)
      outputDir = '../docs/manual/' + os.path.relpath(root, koraliDir)
      outputFile = outputDir + '/' + name
      
      inputDirs.append(inputDir)
      inputFiles.append(inputFile)
      outputDirs.append(outputDir)
      outputFiles.append(outputFile)
      
 # Creating directory
 for dir in outputDirs:
  if (not os.path.isdir(dir)): os.makedirs(dir)
  
 # Detecting modules
 for i in range(len(inputFiles)):
   
   moduleName = os.path.basename(os.path.normpath(inputDirs[i]))
   
   # Loading JSON Configuration
   moduleJsonFile = inputFiles[i].replace('README.md', moduleName + '.config')
   print('[Korali] Opening ' + moduleJsonFile + ' ...')
   
   with open(moduleJsonFile, 'r') as file: moduleJsonString = file.read()
   moduleConfig = json.loads(moduleJsonString)
   
   ####### Producing module page
 
   with open(inputFiles[i], 'r') as file: moduleDocString = file.read()
   
   moduleDocString = '# ' + moduleName.capitalize() + '\n\n' + moduleDocString 
   
   moduleSettingsString = '### Module Configuration\n\n'
   if ('Configuration Settings' in moduleConfig):
    for v in moduleConfig["Configuration Settings"]:
     moduleSettingsString += getVariableInfo(v, moduleName)
    moduleDocString += '### Configuration Settings\n\n'
    moduleDocString += moduleSettingsString
    moduleDocString +='\n\n'  
    
   moduleSiblings = ''
   for root, dirs, files in os.walk(inputDirs[i], topdown=True):
    for dir in dirs:
     if os.path.isfile(os.path.join(root, dir) + '/README.md'):
      moduleSiblings += ' + [' + dir.capitalize() + ']' + '(' + dir + ')\n\n'
    break
   
   if (moduleSiblings != ''):
    moduleDocString += '### Sub-Modules\n\n'
    moduleDocString += moduleSiblings
    moduleDocString +='\n\n'  
     
   print('[Korali] Creating ' + outputFiles[i] + '...')    
   with open(outputFiles[i], 'w+') as file: file.write(moduleDocString)
