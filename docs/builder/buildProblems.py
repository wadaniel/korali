import sys
import os
import json
from buildAux import *

def buildProblems(koraliDir):
 # Processing Problems
 problemsDir = koraliDir + '/source/problems'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/usage/problems/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Initializing Mkdocs entries string
 mkdocsEntriesString = ''
  
 # Detecting Problems
 problemPaths  = os.listdir(problemsDir)
 for problemPath in problemPaths:
  problemPath =  problemsDir + '/' + problemPath
  if (os.path.isdir(problemPath)):
   problemName = problemPath.replace(problemsDir + '/', '')
  
   # Loading JSON Configuration
   problemJsonFile = problemPath + '/' + problemName + '.json'
   if (not os.path.isfile(problemJsonFile)): continue 
   with open(problemJsonFile, 'r') as file: problemJsonString = file.read()
   problemConfig = json.loads(problemJsonString)
   
   # Adding yaml entry
   mkdocsEntriesString += '          - ' + problemConfig["Alias"] + ': usage/problems/' + problemName + '.md\n'
   
   ####### Producing problem page
 
   problemDocFile = problemPath + '/' + problemName + '.md'
   with open(problemDocFile, 'r') as file: problemDocString = file.read()
   
   problemDocString = '# ' + problemConfig["Name"] + '\n\n' + problemDocString 
   
   problemSettingsString = '### Problem Settings\n\n'
   if (not problemConfig["Problem Configuration"]): problemSettingsString += '*none*'
   for v in problemConfig["Problem Configuration"]:
    problemSettingsString += getVariableInfo(v, problemName)
   problemDocString = problemDocString.replace('### Problem Settings', problemSettingsString + '\n\n')  
     
   variableSettingsString = '### Variable-Specific Settings\n\n'
   if (not problemConfig["Variables Configuration"]): variableSettingsString += '*none*'
   for v in problemConfig["Variables Configuration"]: 
     variableSettingsString += getVariableInfo(v, problemName)
   problemDocString = problemDocString.replace('### Variable-Specific Settings', variableSettingsString + '\n\n')
     
   mdFileName = koraliDir + '/docs/docs/usage/problems/' + problemName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(problemDocString)

 return mkdocsEntriesString