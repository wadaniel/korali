#!/usr/bin/env python3

import os

def getDescription(fobj):
 line = fobj.readline()
 description = ''
 while (not line.startswith('*******************************')):
  description += line
  line = fobj.readline()
 return description

def parseFile(f):

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

 with open(f, 'r') as file_object:
  line = file_object.readline()
  while line:
  
   ## Resolving Solver Data
   if (line.startswith('Solver Description')):
    solverName = file_object.replace('Solver Name:', '').strip()
    solverShortName = file_object.replace('Solver Name:', '').strip()
    solverClass = file_object.replace('Class:', '').strip()
    solverType = file_object.readline().replace('Type:', '').strip()
    solverDescription = getDescription(file_object)
     
   ## Resolving Solver Settings
   if (line.startswith('Solver Setting')):
    solverSettingNames.append(file_object.readline().replace('Name:', '').strip())
    solverSettingTypes.append(file_object.readline().replace('Type:', '').strip())
    solverSettingMandatoryFlags.append(file_object.readline().replace('Mandatory:', '').strip())
    solverSettingDefaultValues.append(file_object.readline().replace('Default Value:', '').strip())
    solverSettingDescription.append(getDescription(file_object))
    
   ## Resolving Solver Settings
   if (line.startswith('Variable Setting')):
    variableSettingNames.append(file_object.readline().replace('Name:', '').strip())
    variableSettingTypes.append(file_object.readline().replace('Type:', '').strip())
    variableSettingMandatoryFlags.append(file_object.readline().replace('Mandatory:', '').strip())
    variableSettingDefaultValues.append(file_object.readline().replace('Default Value:', '').strip())
    variableSettingDescription.append(getDescription(file_object))
    
   ## Resolving Solver Settings
   if (line.startswith('Termination Criterion')):
    terminationCriteriaNames.append(file_object.readline().replace('Name:', '').strip())
    terminationCriteriaTypes.append(file_object.readline().replace('Type:', '').strip())
    terminationCriteriaMandatoryFlags.append(file_object.readline().replace('Mandatory:', '').strip())
    terminationCriteriaDefaultValues.append(file_object.readline().replace('Default Value:', '').strip())
    terminationCriteriaDefaultStates.append(file_object.readline().replace('Default State:', '').strip())
    terminationCriteriaDescriptions.append(getDescription(file_object))
          
   line = file_object.readline()
 
 # Print Results
  for i in range(len(solverSettingNames)):
   print(solverSettingNames[i])
   print(solverSettingTypes[i])
   print(solverSettingMandatoryFlags[i])
   print(solverSettingDefaultValues[i])
   print(solverSettingDescription[i])
   
def main():
 # Finding Solver Header Files
 path = '../include/solvers'

 files = []
 for r, d, f in os.walk(path):
  for file in f:
   if '.h' in file:
    files.append(os.path.join(r, file))

 # Parse each file
 for f in files:
  print('Processing: ' + f + '...')
  parseFile(f) 
  
main()
   
   
   
   
   
   