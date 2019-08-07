import sys
import os
import json
from buildAux import *

def buildDistributions(koraliDir):
 # Processing Distributions
 distributionsDir = koraliDir + '/source/distributions'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/manual/distributions/'
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
   if (not os.path.isfile(distributionJsonFile)): continue 
   with open(distributionJsonFile, 'r') as file: distributionJsonString = file.read()
   distributionConfig = json.loads(distributionJsonString)
   
   ####### Producing distribution page
 
   distributionDocFile = distributionPath + '/README.md'
   with open(distributionDocFile, 'r') as file: distributionDocString = file.read()
   
   distributionDocString = '# ' + distributionConfig["Name"] + '\n\n' + distributionDocString 
   
   distributionSettingsString = '### Conditional Variables\n\n'
   if (not distributionConfig["Distribution Configuration"]): distributionSettingsString += '*none*'
   for v in distributionConfig["Distribution Configuration"]:
    distributionSettingsString += getVariableInfo(v, distributionName)
   distributionDocString = distributionDocString.replace('### Distribution Settings', distributionSettingsString + '\n\n')  
     
   mdFileName = koraliDir + '/docs/docs/manual/distributions/' + distributionName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(distributionDocString)

   ####### Adding model list entry
   
   distributionListString += '+ [' + distributionConfig["Name"] + '](distributions/' + distributionName + ')\n'
   
 ###### Saving distribution list

 print('[Korali] Creating distributionList.txt ...')  
 with open('distributionList.txt', 'w+') as file: file.write(distributionListString)
