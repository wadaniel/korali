#!/usr/bin/env python3

import os
configFile = open("config.cpp","w+")

def getDescription(fobj):
 line = fobj.readline()
 line = ''
 description = ''
 while (not line.startswith('*******************************')):
  description += line
  line = fobj.readline()
 return description

################################################################################################

def consumeValue(base, varName, type, default, path = []):
 
 configFile.write('\n')
 
 configFile.write(' if (isDefined(' + base + ', {"' + path[0] + '"')
 for i in range(1, len(path)): configFile.write(', "' + path[i] +'"')
 configFile.write('})) \n { \n')
 
 configFile.write('  ' + varName + ' = ' + base )
 for i in range(0, len(path)-1): configFile.write('.at("' + path[i] +'")')
 getLine = '.at("' + path[-1] + '").get<' + type + '>();\n'
 if ('bool' in type): getLine = '.at("' + path[-1] + '").get<int>() != 0;\n' 
 configFile.write(getLine)
 
 configFile.write('  ' + base)
 for i in range(0, len(path)-1): configFile.write('.at("' + path[i] +'")')
 configFile.write('.erase("' + path[-1] + '");\n')
 
 configFile.write(' }\n else ')
 if (default == ''):
  configFile.write(' { fprintf(stderr, "[Korali] Error: No value provided for mandatory setting: ')
  for i in range(0, len(path)-1): configFile.write('[' + path[i] + '] > ')
  configFile.write('[' + path[-1] + ']\\n"); exit(-1); } ')
 else:
  defaultLine = varName + ' = ' + default + ' ;'
  if ('std::vector<' in type):
   defaultLine = 'for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i] = ' + default + ';'
  if ('std::vector<std::vector<' in type):
   defaultLine = 'for(size_t i = 0; i < ' + varName + '.size(); i++) ' + 'for(size_t j = 0; j < ' + varName + '[i].size(); j++) ' + varName + '[i][j] = ' + default + ';'
  configFile.write(defaultLine)
   
 configFile.write('\n')

#############################################################################################################

def writeConfig(solverName, solverType, solverAlias, solverDescription, solverPlottingDescription, settingNames = [],
                settingTypes = [], settingDefaultValues = [], settingDefaultStates = [],
                settingDescriptions = [], settingVariableNames = [], settingStateNames = [],
                settingDataTypes = []):
  # Creating setConfiguration()

 if (solverName == ''): return

 configFile.write('void Korali::Solver::' + solverAlias + '::setConfiguration() \n{\n')

 ## Load Solver Settings
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Solver Setting'):
   consumeValue('_k->_js', settingVariableNames[i], settingDataTypes[i], settingDefaultValues[i], [ solverAlias, settingNames[i] ])
 
 ## Load Variable Settings
 configFile.write('\n')
 
 configFile.write(' _variableSettings.resize(_k->N);\n') 
 configFile.write(' for(size_t i = 0; i < _k->N; i++) \n { \n')
  
 for i in range(len(settingNames)): 
  if (settingTypes[i] == 'Variable Setting'):
   consumeValue('_k->_js["Variables"][i]', '_variableSettings[i].' + settingVariableNames[i], settingDataTypes[i], settingDefaultValues[i], [ solverAlias, settingNames[i] ])
 configFile.write(' } \n\n')
 
 ## Load Termination Criteria
 
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Termination Criterion'):
   consumeValue('_k->_js', settingVariableNames[i], settingDataTypes[i], settingDefaultValues[i], [ solverAlias, 'Termination Criteria', settingNames[i], 'Value' ])
   consumeValue('_k->_js', settingStateNames[i], 'bool', settingDefaultStates[i], [ solverAlias, 'Termination Criteria', settingNames[i], 'Enabled' ]) 
 configFile.write('\n')
 
 ## Load Solver Internal Attributes
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Internal Attribute'):
   configFile.write(' if(isDefined(_k->_js, {"' + solverAlias + '", "Internal", "' + settingNames[i] + '"} )) \n {\n')
   configFile.write('  ' + settingVariableNames[i] + ' = _k->_js.at("' + solverAlias + '").at("Internal").at("' + settingNames[i] + '").get<' + settingDataTypes[i] + '>();\n')
   configFile.write('  _k->_js["' + solverAlias + '"]["Internal"].erase("' + settingNames[i] + '"); \n }\n\n')
 
 configFile.write('} \n\n') 
 
 # Creating getConfiguration()
 
 configFile.write('void Korali::Solver::' + solverAlias + '::getConfiguration() \n{\n\n')

 configFile.write(' _k->_js["Solver"] = "' + solverAlias + '";\n')

 ## Save Solver Settings
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Solver Setting'):
   configFile.write(' _k->_js["' + solverAlias + '"]["' + settingNames[i] + '"] = ' + settingVariableNames[i] + ';\n')
 
 ## Save Variable Settings
 configFile.write('\n\n for(size_t i = 0; i < _k->N; i++) \n { \n')
  
 for i in range(len(settingNames)): 
  if (settingTypes[i] == 'Variable Setting'):
   configFile.write('  _k->_js["Variables"][i]["' + solverAlias +'"]["' + settingNames[i] + '"] = _variableSettings[i].' + settingVariableNames[i] + ';\n')
   
 configFile.write(' } \n\n')
 
 ## Save Termination Criteria
 
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Termination Criterion'):
   configFile.write(' _k->_js["' + solverAlias + '"]["Termination Criteria"]["' + settingNames[i] + '"]["Value"] = ' + settingVariableNames[i] + ';\n')
   configFile.write(' _k->_js["' + solverAlias + '"]["Termination Criteria"]["' + settingNames[i] + '"]["Enabled"] = ' + settingStateNames[i] + ';\n') 
 
 configFile.write('\n')
 
 ## Save Solver Internal Attributes

 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Internal Attribute'):
   configFile.write(' _k->_js["' + solverAlias + '"]["Internal"]["' + settingNames[i] + '"] = ' + settingVariableNames[i] + ';\n')
  
 configFile.write('} \n\n')
 
###############################################################################################################
 
def writeWeb(solverName, solverType, solverAlias, solverDescription, solverPlottingDescription, settingNames = [],
               settingTypes = [], settingDefaultValues = [], settingDefaultStates = [],
               settingDescriptions = [], settingVariableNames = [], settingStateNames = [],
               settingDataTypes = []):

 import io
 if (solverName == ''): return
 webFile = open( '../docs/docs/usage/solvers/' + solverAlias.lower() + '.md', 'w+')

 webFile.write('# ' + solverName + ' \n\n')
 webFile.write('## Description\n\n')
 webFile.write(solverDescription + '\n\n')
 webFile.write('## Solver Settings\n\n')
 
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Solver Setting'):
   webFile.write('??? abstract "' + settingNames[i] + '"\n\n')
   s = io.StringIO(settingDescriptions[i])
   for line in s:  webFile.write('\t' + line)
   webFile.write('\n')
   webFile.write('\t+ Default Value: ' + settingDefaultValues[i] + '\n')
   webFile.write('\t+ Datatype: ' + settingDataTypes[i] + '\n')
   webFile.write('\t+ Syntax: \n\n')
   webFile.write('\t```python\n\t\t_k->_js["' + solverAlias + '"]["' + settingNames[i] + '"]\n\t```\n\n')
  
 webFile.write('## Variable Settings\n\n')
 
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Variable Setting'):
   webFile.write('??? abstract "' + settingNames[i] + '"\n\n')
   s = io.StringIO(settingDescriptions[i])
   for line in s:  webFile.write('\t' + line)
   webFile.write('\n')
   webFile.write('\t+ Default Value: ' + settingDefaultValues[i] + '\n')
   webFile.write('\t+ Datatype: ' + settingDataTypes[i] + '\n')
   webFile.write('\t+ Syntax: \n\n')
   webFile.write('\t```python\n\t\t_k->_js["Variables"][i]["' + solverAlias + '"]["' + settingNames[i] + '"]\n\t```\n\n')

 webFile.write('## Termination Criteria\n\n')
 
 for i in range(len(settingNames)):   
  if (settingTypes[i] == 'Termination Criterion'):
   webFile.write('??? abstract "' + settingNames[i] + '"\n\n')
   s = io.StringIO(settingDescriptions[i])
   for line in s:  webFile.write('\t' + line)
   webFile.write('\n')
   webFile.write('\t+ Default Value: ' + settingDefaultValues[i] + '\n')
   webFile.write('\t+ Enabled by Default?: ' + settingDefaultStates[i] + '\n')
   webFile.write('\t+ Datatype: ' + settingDataTypes[i] + '\n')
   webFile.write('\t+ Syntax: \n\n')
   webFile.write('\t```python\n\t\t_k->_js["' + solverAlias + '"]["Termination Criteria"]["' + settingNames[i] + '"]["Value"]\n\t```\n\n')
   webFile.write('\t```python\n\t\t_k->_js["' + solverAlias + '"]["Termination Criteria"]["' + settingNames[i] + '"]["Enabled"]\n\t```\n\n')
   
 webFile.write('## Plotting\n\n')
 
 webFile.write(solverPlottingDescription + '\n\n')
   
 webFile.close()
 
################################################################################################################

def parseFile(f):

 solverName = ''
 solverType = ''
 solverAlias = ''
 solverDescription = ''
 solverPlottingDescription = ''
    
 settingNames = []
 settingTypes = []
 settingDefaultValues = []
 settingDefaultStates = []
 settingDescriptions = []
 settingPlottings = []
 settingVariableNames = []
 settingStateNames = []
 settingDataTypes = []

 with open(f, 'r') as file:
  line = file.readline()
  while line:
  
   ## Resolving Solver Data
   if (line.startswith('Module Name:')):
    solverName = line.replace('Module Name:', '').strip()
    solverType = file.readline().replace('Type:', '').strip()
    solverAlias = file.readline().replace('Alias:', '').strip()
    solverDescription = getDescription(file)
    solverPlottingDescription = getDescription(file)
     
   ## Resolving Solver Settings
   if (line.startswith('Setting Name:')):
    settingNames.append(line.replace('Setting Name:', '').strip())
    settingTypes.append(file.readline().replace('Type:', '').strip())
    settingDefaultValues.append(file.readline().replace('Default Value:', '').strip())
    settingDefaultStates.append(file.readline().replace('Default Enabled:', '').strip())
    settingDescriptions.append(getDescription(file))
    declarationWords = file.readline().strip().replace(';', '').split() 
    settingDataTypes.append(declarationWords[0])
    settingVariableNames.append(declarationWords[-1])
    
    stateWords = file.readline().strip().split()
    stateName = ''
    if (len(stateWords) > 0): stateName = stateWords[1] 
    settingStateNames.append(stateName.replace(';', ''))
    
   line = file.readline()
   
 writeConfig(solverName, solverType, solverAlias, solverDescription, solverPlottingDescription, settingNames,
             settingTypes, settingDefaultValues, settingDefaultStates, settingDescriptions, 
             settingVariableNames, settingStateNames, settingDataTypes)
  
 writeWeb(solverName, solverType, solverAlias, solverDescription, solverPlottingDescription, settingNames,
           settingTypes, settingDefaultValues, settingDefaultStates, settingDescriptions, 
           settingVariableNames, settingStateNames, settingDataTypes)
 
####################################################################################################

# Initializing Config File

configFile.write('#include "korali.h"\n')
configFile.write('#include <stdio.h>\n\n')

# Finding Solver Header Files
path = '../include/solvers'
for r, d, f in os.walk(path):
 for filePath in f:
   if filePath.endswith('.h'):
    print('Processing: ' + filePath + '...')
    parseFile(os.path.join(r, filePath)) 
    
configFile.close()    
   
   