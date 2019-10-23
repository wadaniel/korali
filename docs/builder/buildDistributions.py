import sys
import os
import json
from buildAux import *

def buildDistributions(koraliDir):
 # Processing Distributions
 distributionsDir = koraliDir + '/source/distribution'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/manual/distribution/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Creating distribution list string
 distributionListString = ''
 
 # Detecting Distributions
 distributionPaths  = os.listdir(distributionsDir)
 for distributionPath in distributionPaths:
  distributionPath =  distributionsDir + '/' + distributionPath
  if (os.path.isdir(distributionPath)):
   distributionName = distributionPath.replace(distributionsDir + '/', '')
   
   # Loading JSON Configuration
   distributionJsonFile = distributionPath + '/' + distributionName + '.json'
   print('[Korali] Opening ' + distributionJsonFile + '...')
   if (not os.path.isfile(distributionJsonFile)): continue 
   with open(distributionJsonFile, 'r') as file: distributionJsonString = file.read()
   distributionConfig = json.loads(distributionJsonString)
   
   ####### Producing distribution page
 
   distributionDocFile = distributionPath + '/' + distributionName + '.md'
   with open(distributionDocFile, 'r') as file: distributionDocString = file.read()
   
   distributionDocString = '# ' + 'NameHere' + '\n\n' + distributionDocString 
   
   distributionSettingsString = '### Conditional Variables\n\n'
   if (not distributionConfig["Configuration Settings"]): distributionSettingsString += '*none*'
   for v in distributionConfig["Configuration Settings"]:
    distributionSettingsString += getVariableInfo(v, distributionName)
   distributionDocString = distributionDocString.replace('### Distribution Settings', distributionSettingsString + '\n\n')  
     
   mdFileName = koraliDir + '/docs/docs/manual/distribution/' + distributionName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(distributionDocString)

   ####### Adding model list entry
   
   distributionListString += '\t+ [' + distributionConfig["Name"] + '](../distribution/' + distributionName + ')\n'
   
 ###### Saving distribution list

 print('[Korali] Creating distributionList.txt ...')  
 with open('distributionList.txt', 'w+') as file: file.write(distributionListString)
