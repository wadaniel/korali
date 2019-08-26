import sys
import os
import json
from buildAux import *

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) 

##################### Loading Korali Template code files

koraliTemplateHeaderFile = koraliDir + '/korali._hpp'
koraliNewHeaderFile = koraliDir + '/korali.hpp'
with open(koraliTemplateHeaderFile, 'r') as file: newHeaderString = file.read()

koraliTemplateCodeFile = koraliDir + '/korali._cpp'
koraliNewCodeFile = koraliDir + '/korali.cpp'
with open(koraliTemplateCodeFile, 'r') as file: newCodeString = file.read()

koraliJsonPath = koraliDir + '/korali.json'
with open(koraliJsonPath, 'r') as file: koraliJsonObject = json.load(file)
 
##### Processing Korali includes

headerFilesString= ''
for root, dirs, files in os.walk(koraliDir):
 for file in files:
  if file.endswith(".hpp"):
   relPath = os.path.relpath(root, koraliDir)
   headerFilesString += '#include "' + os.path.join(relPath, file) + '"\n'

##### Completing Configuration

newHeaderString = newHeaderString.replace('// Include Files', headerFilesString)
newHeaderString = newHeaderString.replace(' public:', ' public:\n' + createHeaderDeclarations(koraliJsonObject))  
newCodeString += createSetConfiguration(koraliJsonObject)
newCodeString += createGetConfiguration(koraliJsonObject)

##################### Saving output files
  
print('[Korali] Creating: ' + koraliNewHeaderFile + '...')
with open(koraliNewHeaderFile, 'w') as file: file.write(newHeaderString)

print('[Korali] Creating: ' + koraliNewCodeFile + '...')
with open(koraliNewCodeFile, 'w') as file: file.write(newCodeString)
 
 
