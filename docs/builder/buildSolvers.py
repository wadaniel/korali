import sys
import os
import json
from buildAux import *

def buildSolvers(koraliDir):
 # Processing Solvers
 solversDir = koraliDir + '/source/solvers'
 
 # Creating directory
 outputDir = koraliDir + '/docs/docs/manual/solvers/'
 if (not os.path.isdir(outputDir)): os.makedirs(outputDir)
 
 # Creating solver list string
 solverListString = ''
 
 # Detecting Solvers
 solverPaths  = os.listdir(solversDir)
 for solverPath in solverPaths:
  solverPath =  solversDir + '/' + solverPath
  if (os.path.isdir(solverPath)):
   solverName = solverPath.replace(solversDir + '/', '')
  
   # Loading JSON Configuration
   solverJsonFile = solverPath + '/' + solverName + '.json'
   if (not os.path.isfile(solverJsonFile)): continue 
   with open(solverJsonFile, 'r') as file: solverJsonString = file.read()
   solverConfig = json.loads(solverJsonString)
   
   ####### Producing solver page
 
   solverDocFile = solverPath + '/' + solverName + '.md'
   with open(solverDocFile, 'r') as file: solverDocString = file.read()
   
   solverDocString = '# ' + solverConfig["Name"] + '\n\n' + solverDocString 
   
   solverSettingsString = '### Configuration Settings\n\n'
   if (not "Configuration Settings" in solverConfig): solverSettingsString += '*none*'
   else:
    for v in solverConfig["Configuration Settings"]:
     solverSettingsString += getVariableInfo(v, solverName)
   solverDocString = solverDocString.replace('### Solver Settings', solverSettingsString + '\n\n')  
     
   variableSettingsString = '### Variable-Specific Settings\n\n'
   if (not "Variables Configuration" in solverConfig): variableSettingsString += '*none*'
   else:
    for v in solverConfig["Variables Configuration"]: 
     variableSettingsString += getVariableInfo(v, solverName)
   solverDocString = solverDocString.replace('### Variable-Specific Settings', variableSettingsString + '\n\n')
     
   terminationSettingsString = '### Termination Criteria\n\n'
   if (not "Termination Criteria" in solverConfig): terminationSettingsString += '*none*'
   else:
    for v in solverConfig["Termination Criteria"]: 
     terminationSettingsString += getVariableInfo(v, solverName)
   solverDocString = solverDocString.replace('### Termination Criteria', terminationSettingsString + '\n\n')
   
   mdFileName = koraliDir + '/docs/docs/manual/solvers/' + solverName + '.md'
   print('[Korali] Creating ' + mdFileName + '...')    
   with open(mdFileName, 'w+') as file: file.write(solverDocString)

   ####### Adding model list entry
   
   solverListString += '+ [' + solverConfig["Name"] + '](solvers/' + solverName + ')\n'
   
 ###### Returning problem list
 
 return solverListString
