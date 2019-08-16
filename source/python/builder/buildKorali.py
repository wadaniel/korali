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
    conduitDetectString += ' if (_conduitType == "' + conduitConfig['Alias'] + '")  _conduit = new Korali::Conduit::' + conduitConfig['Class'] + '();\n' 
 
 newCodeString = newCodeString.replace(' // Configuring Conduit', conduitDetectString)
 
 #### Parsing Problems

 problemDetectString= ''
 for root, dirs, files in os.walk(koraliDir + '/problems/'):
  for file in files:
   if file.endswith(".json"):
    filePath = os.path.join(root, file)
    with open(filePath, 'r') as file: problemConfig = json.load(file)
    problemDetectString += ' if (_problemType == "' + problemConfig['Alias'] + '")  _problem = new Korali::Problem::' + problemConfig['Class'] + '();\n' 
 
 newCodeString = newCodeString.replace(' // Configuring Problem', problemDetectString)
 
 #### Parsing Solvers

 solverDetectString= ''
 for root, dirs, files in os.walk(koraliDir + '/solvers/'):
  for file in files:
   if file.endswith(".json"):
    filePath = os.path.join(root, file)
    with open(filePath, 'r') as file: solverConfig = json.load(file)
    solverDetectString += ' if (_solverType == "' + solverConfig['Alias'] + '")  _solver = new Korali::Solver::' + solverConfig['Class'] + '();\n' 
 
 newCodeString = newCodeString.replace(' // Configuring Solver', solverDetectString)
   
 ##### Saving Output Files
   
 print('[Korali] Creating: ' + koraliNewHeaderFile + '...')
 with open(koraliNewHeaderFile, 'w') as file: file.write(newHeaderString)
 
 print('[Korali] Creating: ' + koraliNewCodeFile + '...')
 with open(koraliNewCodeFile, 'w') as file: file.write(newCodeString)
 
 
 
 
