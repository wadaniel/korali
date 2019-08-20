import sys
import os
import json
from buildAux import *

def buildKorali(koraliDir):
 koraliTemplateHeaderFile = koraliDir + '/korali._hpp'
 koraliNewHeaderFile = koraliDir + '/korali.hpp'
 with open(koraliTemplateHeaderFile, 'r') as file: newHeaderString = file.read()
 
 koraliTemplateCodeFile = koraliDir + '/korali._cpp'
 koraliNewCodeFile = koraliDir + '/korali.cpp'
 with open(koraliTemplateCodeFile, 'r') as file: newCodeString = file.read()
  
 # Loading JSON Configuration
 koraliJsonFile = koraliDir + '/korali.json'
 with open(koraliJsonFile, 'r') as file: koraliJsonString = file.read()
 koraliConfig = json.loads(koraliJsonString) 
 
 ##### Processing header file
 
 headerFilesString= ''
 for root, dirs, files in os.walk(koraliDir):
  for file in files:
   if file.endswith(".hpp"):
    relPath = os.path.relpath(root, koraliDir)
    headerFilesString += '#include "' + os.path.join(relPath, file) + '"\n'

 newHeaderString = newHeaderString.replace('// Include Files', headerFilesString)
 
 ##### Processing code file
 
 #### Parsing Conduits

 conduitDetectString= ''
 for root, dirs, files in os.walk(koraliDir + '/conduits/'):
  for file in files:
   if file.endswith(".json"):
    filePath = os.path.join(root, file)
    with open(filePath, 'r') as file: conduitConfig = json.load(file)
    conduitDetectString += ' if (_conduitType == "' + conduitConfig['Alias'] + '") { _conduit = new Korali::Conduit::' + conduitConfig['Class'] + '(); recognizedConduit = true; }\n' 
 
 newCodeString = newCodeString.replace(' // Configuring Conduit', conduitDetectString)
 
 #### Parsing Problems

 problemDetectString= ''
 for root, dirs, files in os.walk(koraliDir + '/problems/'):
  for file in files:
   if file.endswith(".json"):
    filePath = os.path.join(root, file)
    with open(filePath, 'r') as file: problemConfig = json.load(file)
    problemDetectString += ' if (_problemType == "' + problemConfig['Alias'] + '") { _problem = new Korali::Problem::' + problemConfig['C++ Class'] + '(); recognizedProblem = true; }\n' 
 
 newCodeString = newCodeString.replace(' // Configuring Problem', problemDetectString)
 
 #### Parsing Solvers

 solverDetectString= ''
 for root, dirs, files in os.walk(koraliDir + '/solvers/'):
  for file in files:
   if file.endswith(".json"):
    filePath = os.path.join(root, file)
    with open(filePath, 'r') as file: solverConfig = json.load(file)
    solverDetectString += ' if (_solverType == "' + solverConfig['Alias'] + '") { _solver = new Korali::Solver::' + solverConfig['C++ Class'] + '(); recognizedSolver = true; }\n' 
 
 newCodeString = newCodeString.replace(' // Configuring Solver', solverDetectString)
   
 ##### Parsing korali.json
 
 koraliJsonFile = koraliDir + '/korali.json'
 with open(koraliJsonFile, 'r') as file: koraliJsonString = file.read()
 koraliConfig = json.loads(koraliJsonString)
  
 ## Producing private variable declarations
 
 koraliHeaderString = ''
  
 for v in koraliConfig["Module Selections"]:
  koraliHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'
     
 for v in koraliConfig["General Settings"]:
  koraliHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'

 for v in koraliConfig["Internal Settings"]:
  koraliHeaderString += getVariableType(v) + ' ' + getCXXVariableName(v) + ';\n'

 newHeaderString = newHeaderString.replace(' public:', ' public:\n' + koraliHeaderString)

 ### Producing getConfiguration
 
 ## Consume Korali Settings
 
 # Detecting Modules
 
 koraliConsumeString = ''
 
 for v in koraliConfig["Module Selections"]:
  koraliConsumeString += consumeValue('_js', 'Korali', getVariablePath(v), getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
  
 newCodeString = newCodeString.replace(' // Detecting Korali Modules', koraliConsumeString) 
  
 # Detecting Settings
  
 koraliConsumeString = ''
 
 for v in koraliConfig["General Settings"]:
  koraliConsumeString += consumeValue('js', 'Korali', '["General"]' + getVariablePath(v),  getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
  
 for v in koraliConfig["Internal Settings"]:
  koraliConsumeString += consumeValue('js', 'Korali', '["Internal"]' + getVariablePath(v),  getCXXVariableName(v), getVariableType(v), getVariableDefault(v))
 
 newCodeString = newCodeString.replace(' // Configuring Korali Engine', koraliConsumeString)
 
 # Saving Korali Settings
 
 koraliGetString = ''
 
 for v in koraliConfig["Module Selections"]: 
  koraliGetString += saveValue('_js', getVariablePath(v), getCXXVariableName(v), getVariableType(v))
 
 for v in koraliConfig["General Settings"]: 
  koraliGetString += saveValue('_js', '["General"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v))
    
 for v in koraliConfig["Internal Settings"]: 
  koraliGetString += saveValue('_js', '["Internal"]' + getVariablePath(v), getCXXVariableName(v), getVariableType(v))
     
 newCodeString = newCodeString.replace(' // Obtaining Korali Engine Settings', koraliGetString)

 ##### Saving output files
   
 print('[Korali] Creating: ' + koraliNewHeaderFile + '...')
 with open(koraliNewHeaderFile, 'w') as file: file.write(newHeaderString)
 
 print('[Korali] Creating: ' + koraliNewCodeFile + '...')
 with open(koraliNewCodeFile, 'w') as file: file.write(newCodeString)
 
 
 
 
