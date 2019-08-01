import sys
import os
import json
from buildAux import *

def buildTutorials(koraliDir):
 # Processing Tutorials
 tutorialsDir = koraliDir + '/tutorials'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/tutorials/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Table Rows Strings
 tableRowsStringDict = { 'A': [], 'B': [], 'C': [], 'D': [] }
 
 # Detecting Tutorials
 tutorialPaths  = os.listdir(tutorialsDir)
 for tutorialPath in tutorialPaths:
  tutorialPath =  tutorialsDir + '/' + tutorialPath
  if (os.path.isdir(tutorialPath)):
   tutorialName = tutorialPath.replace(tutorialsDir + '/', '')
   
   ###### Loading JSON Configuration
   
   tutorialJsonFile = tutorialPath + '/' + tutorialName + '.json'
   if (not os.path.isfile(tutorialJsonFile)): continue 
   with open(tutorialJsonFile, 'r') as file: tutorialJsonString = file.read()
   tutorialConfig = json.loads(tutorialJsonString)
   
   ####### Link to tutorial
   
   tutorialLink = 'tutorials/' + tutorialName + '.md'
   
   ####### Producing Table Entry
   
   tutorialCategory = tutorialConfig["Code"][0]
   rowString  = ' | ' + tutorialConfig["Code"] + ' | [' + tutorialConfig["Title"] + '](' + tutorialLink + ') | \n'
   tableRowsStringDict[tutorialCategory].append(rowString)
   
   ####### Producing tutorial page
   
   tutorialDocFile = tutorialPath + '/README.md'
   with open(tutorialDocFile, 'r') as file: tutorialDocString = file.read()
   
   sourceCodeString = '!!! info\n'
   sourceCodeString += '\tThe **Python** source for this tutorial can be found at: **[' + tutorialName + '/python/](https://github.com/cselab/korali/tree/master/tutorials/' + tutorialName + '/python/)**\n\n'
   sourceCodeString += '\tThe **C++** source for this tutorial can be found at: **[' + tutorialName + '/cxx/](https://github.com/cselab/korali/tree/master/tutorials/' + tutorialName + '/cxx/)**\n\n'
   
   tutorialDocString = tutorialDocString.replace('## Source Code', '## Source Code\n\n' + sourceCodeString)
   
   mdFileName = outputDir + tutorialName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(tutorialDocString)

 ###### Loading tutorials web page

 with open(tutorialsDir + '/README.md', 'r') as file: tutorialWebString = file.read()
   
 ###### Creating Tutorial Tables
 
 for category in tableRowsStringDict:
  tutorialTableString =  " |     |  Tutorial                                    |\n"
  tutorialTableString += " |:---:|:--------------------------------------------:|\n"
  for row in tableRowsStringDict[category]:  tutorialTableString += row
  tutorialWebString = tutorialWebString.replace('<!--- Tutorials ' + category + ' List --->', tutorialTableString)
  
 ###### Saving tutorials web page
 
 webFileName = koraliDir + '/docs/docs/tutorials.md'
 print('[Korali] Creating ' + webFileName + '...')  
 with open(webFileName, 'w+') as file: file.write(tutorialWebString)