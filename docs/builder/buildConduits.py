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
   
   conduitSettingsString = '### Conduit Settings\n\n'
   if (not conduitConfig["Conduit Configuration"]): conduitSettingsString += '*none*'
   for v in conduitConfig["Conduit Configuration"]:
    conduitSettingsString += getVariableInfo(v, conduitName)
   conduitDocString = conduitDocString.replace('### Conduit Settings', conduitSettingsString + '\n\n')  
     
   mdFileName = koraliDir + '/docs/docs/manual/conduits/' + conduitName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(conduitDocString)

   ####### Adding model list entry
   
   conduitListString += '+ [' + conduitConfig["Name"] + '](' + mdFileName + ')\n'
   
 ###### Returning problem list
 
 return conduitListString
