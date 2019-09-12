import sys
import os
import json
from buildAux import *

def buildConduits(koraliDir):
 # Processing Conduits
 conduitsDir = koraliDir + '/source/conduits'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/manual/conduits/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Creating conduit list string
 conduitListString = ''
 
 # Detecting Conduits
 conduitPaths  = os.listdir(conduitsDir)
 for conduitPath in conduitPaths:
  conduitPath =  conduitsDir + '/' + conduitPath
  if (os.path.isdir(conduitPath)):
   conduitName = conduitPath.replace(conduitsDir + '/', '')
  
   # Loading JSON Configuration
   conduitJsonFile = conduitPath + '/' + conduitName + '.json'
   if (not os.path.isfile(conduitJsonFile)): continue 
   with open(conduitJsonFile, 'r') as file: conduitJsonString = file.read()
   conduitConfig = json.loads(conduitJsonString)
   
   ####### Producing conduit page
 
   conduitDocFile = conduitPath + '/README.md'
   with open(conduitDocFile, 'r') as file: conduitDocString = file.read()
   
   conduitDocString = '# ' + conduitConfig["Name"] + '\n\n' + conduitDocString 
   
   conduitSettingsString = '### Configuration Settings\n\n'
   if (not "Configuration Settings" in conduitConfig): conduitSettingsString += '*none*'
   else:
    for v in conduitConfig["Configuration Settings"]:
     conduitSettingsString += getVariableInfo(v, conduitName)
   conduitDocString = conduitDocString.replace('### Conduit Settings', conduitSettingsString + '\n\n')  
     
   variableSettingsString = '### Variable-Specific Settings\n\n'
   if (not "Variables Configuration" in conduitConfig): variableSettingsString += '*none*'
   else:
    for v in conduitConfig["Variables Configuration"]: 
     variableSettingsString += getVariableInfo(v, conduitName)
   conduitDocString = conduitDocString.replace('### Variable-Specific Settings', variableSettingsString + '\n\n')
     
   terminationSettingsString = '### Termination Criteria\n\n'
   if (not "Termination Criteria" in conduitConfig): terminationSettingsString += '*none*'
   else:
    for v in conduitConfig["Termination Criteria"]: 
     terminationSettingsString += getVariableInfo(v, conduitName)
   conduitDocString = conduitDocString.replace('### Termination Criteria', terminationSettingsString + '\n\n')
     
   mdFileName = koraliDir + '/docs/docs/manual/conduits/' + conduitName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(conduitDocString)

   ####### Adding model list entry
   
   conduitListString += '+ [' + conduitConfig["Name"] + '](conduits/' + conduitName + ')\n'
   
 ###### Returning conduit list
 
 return conduitListString
