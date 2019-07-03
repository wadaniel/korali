#!/usr/bin/env python3

import os

configFile = open("config.cpp","w+")

def getDescription(fobj):
 line = fobj.readline()
 description = ''
 while (not line.startswith('*******************************')):
  description += line
  line = fobj.readline()
 return description

def parseFile(f):

 solverName = ''
 solverShortName = ''
 solverClass = ''
 solverType = ''
 solverDescription = ''
    
 solverSettingNames = []
 solverSettingTypes = []
 solverSettingMandatoryFlags = []
 solverSettingDefaultValues = []
 solverSettingDescription = []
 
 variableSettingNames = []
 variableSettingTypes = []
 variableSettingMandatoryFlags = []
 variableSettingDefaultValues = []
 variableSettingDescription = []
 
 terminationCriteriaNames = []
 terminationCriteriaTypes = []
 terminationCriteriaMandatoryFlags = []
 terminationCriteriaDefaultValues = []
 terminationCriteriaDefaultStates = []
 terminationCriteriaDescriptions = []

 with open(f, 'r') as file:
  line = file.readline()
  while line:
  
   ## Resolving Solver Data
   if (line.startswith('Solver Description')):
    solverName = file.readline().replace('Name:', '').strip()
    solverShortName = file.readline().replace('Short Name:', '').strip()
    solverClass = file.readline().replace('Class:', '').strip()
    solverType = file.readline().replace('Type:', '').strip()
    solverDescription = getDescription(file)
     
   ## Resolving Solver Settings
   if (line.startswith('Solver Setting')):
    solverSettingNames.append(file.readline().replace('Name:', '').strip())
    solverSettingTypes.append(file.readline().replace('Type:', '').strip())
    solverSettingMandatoryFlags.append(file.readline().replace('Mandatory:', '').strip())
    solverSettingDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    solverSettingDescription.append(getDescription(file))
    
   ## Resolving Solver Settings
   if (line.startswith('Variable Setting')):
    variableSettingNames.append(file.readline().replace('Name:', '').strip())
    variableSettingTypes.append(file.readline().replace('Type:', '').strip())
    variableSettingMandatoryFlags.append(file.readline().replace('Mandatory:', '').strip())
    variableSettingDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    variableSettingDescription.append(getDescription(file))
    
   ## Resolving Solver Settings
   if (line.startswith('Termination Criterion')):
    terminationCriteriaNames.append(file.readline().replace('Name:', '').strip())
    terminationCriteriaTypes.append(file.readline().replace('Type:', '').strip())
    terminationCriteriaMandatoryFlags.append(file.readline().replace('Mandatory:', '').strip())
    terminationCriteriaDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    terminationCriteriaDefaultStates.append(file.readline().replace('Default State:', '').strip())
    terminationCriteriaDescriptions.append(getDescription(file))
          
   line = file.readline()
 
 # Save solver configuration
 configFile.write('Korali::' + solverClass + '::setConfiguration() \n{')
 
 #for i in range(len(solverSettingNames)):
  #print(solverSettingNames[i])
  #print(solverSettingTypes[i])
  #print(solverSettingMandatoryFlags[i])
  #print(solverSettingDefaultValues[i])
  #print(solverSettingDescription[i]) 
  
 configFile.write('\n} \n\n') 

# Finding Solver Header Files
path = '../include/solvers'
for r, d, f in os.walk(path):
 for filePath in f:
  if '.h' in filePath:
   if not 'base.h' in filePath:
    print('Processing: ' + filePath + '...')
    parseFile(os.path.join(r, filePath)) 
   
configFile.close()    
   
   