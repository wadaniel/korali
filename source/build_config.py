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
 solverSettingDescriptions = []
 solverSettingDeclarations = []
 
 variableSettingNames = []
 variableSettingTypes = []
 variableSettingMandatoryFlags = []
 variableSettingDefaultValues = []
 variableSettingDescriptions = []
 variableSettingDeclarations = []
 
 terminationCriteriaNames = []
 terminationCriteriaTypes = []
 terminationCriteriaMandatoryFlags = []
 terminationCriteriaDefaultValues = []
 terminationCriteriaDefaultStates = []
 terminationCriteriaDescriptions = []
 terminationCriteriaDeclarations = []

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
    solverSettingDescriptions.append(getDescription(file))
    solverSettingDeclarations.append(file.readline().strip())
    
   ## Resolving Variable Settings
   if (line.startswith('Variable Setting')):
    variableSettingNames.append(file.readline().replace('Name:', '').strip())
    variableSettingTypes.append(file.readline().replace('Type:', '').strip())
    variableSettingMandatoryFlags.append(file.readline().replace('Mandatory:', '').strip())
    variableSettingDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    variableSettingDescriptions.append(getDescription(file))
    variableSettingDeclarations.append(file.readline().strip())
    
   ## Resolving Termination Criteria
   if (line.startswith('Termination Criterion')):
    terminationCriteriaNames.append(file.readline().replace('Name:', '').strip())
    terminationCriteriaTypes.append(file.readline().replace('Type:', '').strip())
    terminationCriteriaMandatoryFlags.append(file.readline().replace('Mandatory:', '').strip())
    terminationCriteriaDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    terminationCriteriaDefaultStates.append(file.readline().replace('Default State:', '').strip())
    terminationCriteriaDescriptions.append(getDescription(file))
    terminationCriteriaDeclarations.append(file.readline().strip())
          
   line = file.readline()
  
 # Creating setConfiguration()

 configFile.write('void Korali::Solver::' + solverClass + '::setConfiguration() \n{\n\n')
 configFile.write('printf("Setting ' + solverClass + ' configuration. \\n");\n\n')
 
 for i in range(len(solverSettingNames)):
 
  if (solverSettingTypes[i] == 'Integer'):
   koraliVarType = 'KORALI_NUMBER'
   varName = solverSettingDeclarations[i].replace('size_t', '').replace(';', '').strip()
   
  if (solverSettingTypes[i] == 'Real'):
   koraliVarType = 'KORALI_NUMBER'
   varName = solverSettingDeclarations[i].replace('double', '').replace(';', '').strip()
  
  if (solverSettingTypes[i] == 'String'):
   koraliVarType = 'KORALI_STRING'
   varName = solverSettingDeclarations[i].replace('std::string', '').replace(';', '').strip()
  
  if (solverSettingTypes[i] == 'Boolean'):
   koraliVarType = 'KORALI_BOOLEAN'
   varName = solverSettingDeclarations[i].replace('bool', '').replace(';', '').strip()
  
  configFile.write(varName + ' =  consume(_k->_js, { "')
  
  configFile.write(solverShortName + '", "' + solverSettingNames[i] + '" }, ' + koraliVarType)
  
  if (solverSettingDefaultValues[i] != ''): 
   configFile.write(', "' + solverSettingDefaultValues[i] + '"' )
  
  configFile.write('); \n')
  
 configFile.write('\n} \n\n') 
 
 # Creating getConfiguration()
 
 configFile.write('void Korali::Solver::' + solverClass + '::getConfiguration() \n{\n')
 configFile.write('printf("Getting ' + solverClass + ' configuration. \\n");\n')
 configFile.write('\n} \n\n') 

# Initializing Config File

configFile.write('#include "korali.h"\n\n')

# Finding Solver Header Files
path = '../include/solvers'
for r, d, f in os.walk(path):
 for filePath in f:
  if '.h' in filePath:
   if not 'base.h' in filePath:
    print('Processing: ' + filePath + '...')
    parseFile(os.path.join(r, filePath)) 
   
configFile.close()    
   
   