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
 solverType = ''
 solverAlias = ''
 solverDescription = ''
    
 settingNames = []
 settingTypes = []
 settingFormats = []
 settingMandatoryFlags = []
 settingDefaultValues = []
 settingDefaultStates = []
 settingDescriptions = []
 settingVariableDeclarations = []
 settingStateDeclarations = []
 settingVariableNames = []
 settingStateNames = []
 settingKoraliDataTypes = []

 with open(f, 'r') as file:
  line = file.readline()
  while line:
  
   ## Resolving Solver Data
   if (line.startswith('Module Name:')):
    solverName = line.replace('Module Name:', '').strip()
    solverType = file.readline().replace('Type:', '').strip()
    solverAlias = file.readline().replace('Alias:', '').strip()
    solverDescription = getDescription(file)
     
   ## Resolving Solver Settings
   if (line.startswith('Setting Name:')):
    settingNames.append(line.replace('Setting Name:', '').strip())
    settingTypes.append(file.readline().replace('Type:', '').strip())
    settingFormats.append(file.readline().replace('Format:', '').strip())
    settingMandatoryFlags.append(file.readline().replace('Mandatory:', '').strip())
    settingDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    settingDefaultStates.append(file.readline().replace('Default Enabled:', '').strip())
    settingDescriptions.append(getDescription(file))
    settingVariableDeclarations.append(file.readline().strip())
    settingStateDeclarations.append(file.readline().strip())
    
   line = file.readline()
  
 # Creating setConfiguration()

 configFile.write('void Korali::Solver::' + solverAlias + '::setConfiguration() \n{\n')
 
 ## Post-processing variable information
 
 for i in range(len(settingNames)):
  settingVariableNames.append(settingVariableDeclarations[i].replace('size_t', '').replace('double', '').replace('std::string', '').replace('bool', '').replace('std::vector', '').replace('<', '').replace('>', '').replace(';', '').strip())
  settingStateNames.append(settingStateDeclarations[i].replace('bool', '').replace(';', '').strip())
  
 for i in range(len(settingNames)): 
  if (settingFormats[i] == 'Integer'):
   settingKoraliDataTypes.append('KORALI_NUMBER')
  if (settingFormats[i] == 'Real'):
   settingKoraliDataTypes.append('KORALI_NUMBER')
  if (settingFormats[i] == 'String'):
   settingKoraliDataTypes.append('KORALI_STRING')
  if (settingFormats[i] == 'Boolean'):
   settingKoraliDataTypes.append('KORALI_BOOLEAN')

 ## Writing Solver Settings
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Solver Setting'):
   configFile.write(' ' + settingVariableNames[i] + ' =  consume(_k->_js, { "')
   configFile.write(solverAlias + '", "' + settingNames[i] + '" }, ' + settingKoraliDataTypes[i])
   if (settingDefaultValues[i] != ''): 
    configFile.write(', "' + settingDefaultValues[i] + '"' )
   configFile.write('); \n')
 
 ## Writing Variable Settings
 configFile.write('\n')
 
 for i in range(len(settingNames)): 
  if (settingTypes[i] == 'Variable Setting'):
   configFile.write(' ' + settingVariableNames[i] + '.reserve(_k->N);\n')
  
 configFile.write('\n for(size_t i = 0; i < _k->N; i++) \n { \n')
  
 for i in range(len(settingNames)): 
  if (settingTypes[i] == 'Variable Setting'):
   configFile.write('  ' + settingVariableNames[i] + '[i] =  consume(_k->_js["Variables"][i]')
   configFile.write(', { "' + solverAlias + '", "' + settingNames[i] + '" }, ' + settingKoraliDataTypes[i])
   if (settingDefaultValues[i] != ''): 
    configFile.write(', "' + settingDefaultValues[i] + '"' )
   configFile.write('); \n')
 
 configFile.write(' } \n\n')
 
 ## Writing Termination Criteria
 
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Termination Criterion'):
   configFile.write(' ' + settingVariableNames[i] + ' =  consume(_k->_js, { "')
   configFile.write(solverAlias + '", "Termination Criteria", "' + settingNames[i] + '", "Value" }, ' + settingKoraliDataTypes[i])
   if (settingDefaultValues[i] != ''): 
    configFile.write(', "' + settingDefaultValues[i] + '" );' )
   configFile.write('\n')
    
   configFile.write(' ' + settingStateNames[i] + ' =  consume(_k->_js, { "')
   configFile.write(solverAlias + '", "Termination Criteria", "' + settingNames[i] + '", "State" }, KORALI_BOOLEAN')
   if (settingDefaultStates[i] != ''): 
    configFile.write(', "' + settingDefaultStates[i] + '" );' )
   configFile.write('\n')
 configFile.write('\n')
 
 configFile.write('} \n\n') 
 
 # Creating getConfiguration()
 
 configFile.write('void Korali::Solver::' + solverAlias + '::getConfiguration() \n{\n')
 configFile.write('printf("Getting ' + solverAlias + ' configuration. \\n");\n')
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
   
   